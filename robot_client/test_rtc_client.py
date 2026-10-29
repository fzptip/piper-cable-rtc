"""Offline-only tests. ROS, cameras, networking and publishers are mocked."""
import collections
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import Mock, patch

import numpy as np


ros = types.ModuleType('rospy')
for name in ('loginfo','logwarn','logerr','signal_shutdown'):
    setattr(ros,name,Mock())
ros.is_shutdown = lambda: False
ros.Publisher = Mock(side_effect=AssertionError('Offline test must never create a ROS publisher'))
ros.init_node = Mock(side_effect=AssertionError('Offline test must never initialize ROS'))
modules = {'rospy':ros}
for name in ('cv2','msgpack','websocket','termios','tty','cv_bridge','sensor_msgs','sensor_msgs.msg','std_msgs','std_msgs.msg'):
    modules[name]=types.ModuleType(name)
modules['cv_bridge'].CvBridge=Mock
modules['sensor_msgs.msg'].Image=object
modules['sensor_msgs.msg'].JointState=object
modules['std_msgs.msg'].Header=object
with patch.dict(sys.modules,modules):
    spec=importlib.util.spec_from_file_location('rtc_client_under_test',Path(__file__).with_name('pi05_ws_client.py'))
    client=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(client)


class Observation:
    def __init__(self): self.stopped=False
    def get_qpos(self): return np.zeros(14,np.float32)
    def get_qpos_with_version(self): return self.get_qpos(),(1,1)
    def stop_capture(self): self.stopped=True
    def start_capture(self): self.stopped=False
    def stop_video_recording(self): pass
    def start_video_recording(self): pass


def build(mode='guided'):
    with patch.object(sys,'argv',['client','--rtc-mode',mode,'--chunk-steps','30']):
        args=client.get_arguments()
    obs=Observation()
    publisher=Mock()
    control=client.ArmController(args,obs,publisher,np.zeros(14),np.ones(14))
    control.state=client.ControlState.POLICY
    control.target=np.zeros(14,np.float32)
    return args,obs,publisher,control


def response(request,offset=0):
    return {'actions':np.tile(np.arange(50,dtype=np.float32)[:,None],(1,14))+offset,
            'rtc_info':{**request,'guidance_applied':'prefix' in request}}


def bootstrap(control):
    req=control.begin_rtc_request(control.epoch)
    control.submit_rtc_result(response(req),control.epoch)


class ClientTests(unittest.TestCase):
    def test_replan_before_empty_and_keep_complete_chunk(self):
        _,_,_,c=build(); bootstrap(c)
        self.assertEqual(c.rtc_queue.remaining,50)
        for _ in range(29): c._next_target()
        self.assertIsNone(c.policy_request())
        c._next_target()
        self.assertEqual(c.policy_request(),c.epoch)
        req=c.begin_rtc_request(c.epoch)
        self.assertEqual(req['observation_tick'],30)
        self.assertEqual(req['prefix'].shape,(20,14))
        self.assertEqual(c.rtc_queue.remaining,20)

    def test_early_response_honors_prefix_commitment(self):
        _,_,_,c=build(); bootstrap(c)
        for _ in range(30): c._next_target()
        req=c.begin_rtc_request(c.epoch)
        for _ in range(3): c._next_target()
        c.submit_rtc_result(response(req,1000),c.epoch)
        for value in (33,34,1005):
            np.testing.assert_array_equal(c._next_target(),np.full(14,value))

    def test_late_response_aligns_to_consumed_tick(self):
        _,_,_,c=build(); bootstrap(c)
        for _ in range(30): c._next_target()
        req=c.begin_rtc_request(c.epoch)
        for _ in range(7): c._next_target()
        c.submit_rtc_result(response(req,1000),c.epoch)
        self.assertTrue(c.rtc_queue.last_merge['budget_overrun'])
        np.testing.assert_array_equal(c._next_target(),np.full(14,1007))

    def test_only_one_inflight(self):
        _,_,_,c=build()
        self.assertIsNotNone(c.begin_rtc_request(c.epoch))
        self.assertIsNone(c.begin_rtc_request(c.epoch))
        self.assertIsNone(c.policy_request())

    def test_i_and_e_cancel_old_responses(self):
        for action in ('request_go_a','request_exit'):
            _,_,_,c=build(); epoch=c.epoch
            req=c.begin_rtc_request(epoch)
            getattr(c,action)()
            self.assertFalse(c.submit_rtc_result(response(req),epoch))
            self.assertEqual(c.rtc_queue.remaining,0)

    def test_discard_first_chunk_releases_pending_request(self):
        _,_,_,c=build()
        first=c.begin_rtc_request(c.epoch)
        self.assertTrue(c.discard_initial_rtc_result(c.epoch))
        second=c.begin_rtc_request(c.epoch)
        self.assertNotEqual(first['session_id'],second['session_id'])
        self.assertNotIn('prefix',second)
        self.assertTrue(c.submit_rtc_result(response(second),c.epoch))

    def test_exhaustion_holds_without_planning_return_motion(self):
        _,obs,pub,c=build(); bootstrap(c)
        for _ in range(50): c._next_target()
        last=c.target.copy(); c._next_target()
        self.assertEqual(c.state,client.ControlState.POLICY_HOLD)
        np.testing.assert_array_equal(c.target,last)
        self.assertIsNone(c.trajectory)
        self.assertTrue(obs.stopped)
        pub.publish.assert_not_called()
        self.assertIsNone(c.policy_request())

    def test_legacy_deque_behavior_is_preserved(self):
        _,_,_,c=build('legacy')
        c.actions.extend([np.ones(14),np.ones(14)*2])
        self.assertIsNone(c.policy_request())
        c._next_target(); c._next_target()
        self.assertEqual(c.policy_request(),c.epoch)

    def test_invalid_or_stale_wire_identity_is_rejected(self):
        _,_,_,c=build(); req=c.begin_rtc_request(c.epoch)
        bad=response(req); bad['rtc_info']['request_id']+=1
        with self.assertRaises(ValueError): c.submit_rtc_result(bad,c.epoch)

    def test_loop_discards_first_response_then_bootstraps(self):
        args,obs,pub,c=build()
        obs.get=lambda request_snapshot=None:{'rtc':request_snapshot()}
        policy=Mock()
        policy.server_metadata={'rtc_protocol':1,'action_horizon':50,'action_dim':14}
        policy.infer.side_effect=lambda o: response(o['rtc'])
        with patch.object(ros,'is_shutdown',side_effect=lambda:policy.infer.call_count>=2):
            client.run_rtc_policy_loop(args,c,obs,policy)
        self.assertTrue(c.rtc_bootstrapped)
        self.assertEqual(c.rtc_queue.remaining,50)
        pub.publish.assert_not_called()

    def test_network_error_pauses_without_returning_to_a(self):
        args,obs,pub,c=build()
        obs.get=lambda request_snapshot=None:{'rtc':request_snapshot()}
        policy=Mock()
        policy.server_metadata={'rtc_protocol':1,'action_horizon':50,'action_dim':14}
        policy.infer.side_effect=RuntimeError('simulated timeout')
        policy.close.side_effect=lambda:self.assertEqual(c.state,client.ControlState.POLICY_HOLD)
        with patch.object(ros,'is_shutdown',side_effect=lambda:policy.infer.call_count>=1):
            client.run_rtc_policy_loop(args,c,obs,policy)
        self.assertIsNone(c.trajectory)
        pub.publish.assert_not_called()

    def test_observation_snapshot_runs_outside_camera_lock(self):
        o=object.__new__(client.RosObservation)
        o.lock=threading.Lock(); o.capture_enabled=True
        o.left=np.zeros(7,np.float32); o.right=np.zeros(7,np.float32)
        o.last_observation_stamp=None
        o.image_queues={k:collections.deque([(10.0,object())]) for k in client.CAMERA_NAMES}
        o.bridge=Mock(); o.bridge.imgmsg_to_cv2.side_effect=lambda *a,**kw:np.zeros((224,224,3),np.uint8)
        def snapshot():
            self.assertFalse(o.lock.locked())
            return {'observation_tick':3}
        with patch.object(client,'resize_with_pad',side_effect=lambda x:x),patch.object(client,'preprocess_wrist',side_effect=lambda x:x):
            result=o.get(request_snapshot=snapshot)
        self.assertEqual(result['rtc']['observation_tick'],3)
        self.assertEqual(result['state'].shape,(14,))

    def test_close_error_does_not_escape_pause_handling(self):
        args,obs,_,c=build()
        obs.get=lambda request_snapshot=None:{'rtc':request_snapshot()}
        policy=Mock()
        policy.server_metadata={'rtc_protocol':1,'action_horizon':50,'action_dim':14}
        policy.infer.side_effect=RuntimeError('simulated timeout')
        policy.close.side_effect=RuntimeError('simulated close error')
        with patch.object(ros,'is_shutdown',side_effect=lambda:policy.infer.call_count>=1):
            client.run_rtc_policy_loop(args,c,obs,policy)
        self.assertEqual(c.state,client.ControlState.POLICY_HOLD)

    def test_unexpected_main_failure_does_not_plan_return_motion(self):
        args,_,_,_=build()
        controller=Mock()
        controller.wait_ready.return_value=True
        controller.error_event.is_set.return_value=False
        controller.at_b_event.is_set.return_value=False
        controller.ready_event.is_set.return_value=True
        controller.exit_requested=False
        with patch.object(client,'get_arguments',return_value=args), \
             patch.object(ros,'init_node'), \
             patch.object(client,'RosObservation'), \
             patch.object(client,'ArmPublisher'), \
             patch.object(client,'ArmController',return_value=controller), \
             patch.object(client,'KeyboardMonitor'), \
             patch.object(client,'PolicyClient'), \
             patch.object(client.signal,'signal'), \
             patch.object(client,'run_rtc_policy_loop',side_effect=RuntimeError('unexpected')):
            with self.assertRaises(RuntimeError): client.main()
        controller.request_exit.assert_not_called()
        controller.pause_rtc.assert_called_once()


if __name__=='__main__': unittest.main(verbosity=2)
