"""
Tiny stand-in for rclpy and the message packages.

It lets the real node classes from this repo run inside a plain Python
script (no ROS install, no Gazebo), so we can test the full
planner -> controller loop on a laptop or in CI.
Only the parts our nodes actually use are implemented.
"""

import sys
import types
from collections import defaultdict

BUS = defaultdict(list)      # topic -> list of callbacks
LAST = {}                    # topic -> last published message
PARAMS = {}                  # parameter overrides, like a params.yaml
SIM = {"t": 0.0}             # simulated time [s]


class Msg(object):
    """Message with auto-created sub-fields (msg.pose.pose.position.x = 1 just works)."""
    _lists = ("waypoints", "points", "markers", "blue_cones", "yellow_cones", "orange_cones",
              "big_orange_cones", "unknown_color_cones")

    def __init__(self, **kw):
        for k in self._lists:
            object.__setattr__(self, k, [])
        for k, v in kw.items():
            object.__setattr__(self, k, v)

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        v = Msg()
        object.__setattr__(self, name, v)
        return v


class Marker(Msg):
    ARROW, CUBE, SPHERE, CYLINDER, LINE_STRIP, LINE_LIST = 0, 1, 2, 3, 4, 5
    POINTS, TEXT_VIEW_FACING = 8, 9
    ADD, DELETE, DELETEALL = 0, 2, 3


class _Publisher(object):
    def __init__(self, topic):
        self.topic = topic

    def publish(self, msg):
        LAST[self.topic] = msg
        for cb in list(BUS[self.topic]):
            cb(msg)


def publish(topic, msg):
    _Publisher(topic).publish(msg)


class _Param(object):
    def __init__(self, value):
        self.value = value


class _Logger(object):
    def __init__(self):
        self.lines = []

    def _log(self, m):
        self.lines.append(str(m))

    info = debug = warn = warning = error = _log


class _Time(object):
    def __init__(self):
        self.nanoseconds = int(SIM["t"] * 1e9)

    def to_msg(self):
        return Msg(sec=int(SIM["t"]), nanosec=int((SIM["t"] % 1) * 1e9))


class _Clock(object):
    def now(self):
        return _Time()


class Node(object):
    def __init__(self, name, **kw):
        self._logger = _Logger()
        self._timers = []

    def declare_parameter(self, name, default=None):
        return _Param(PARAMS.get(name, default))

    def get_parameter(self, name):
        return _Param(PARAMS.get(name))

    def create_subscription(self, msg_type, topic, cb, qos):
        BUS[topic].append(cb)
        return cb

    def create_publisher(self, msg_type, topic, qos):
        return _Publisher(topic)

    def create_timer(self, period, cb):
        self._timers.append([period, cb, 0.0])
        return cb

    def create_service(self, *a, **k):
        return None

    def spin_timers(self):
        """Call timers that are due at the current SIM time."""
        for t in self._timers:
            if SIM["t"] - t[2] >= t[0] - 1e-9:
                t[2] = SIM["t"]
                t[1]()

    def get_logger(self):
        return self._logger

    def get_clock(self):
        return _Clock()

    def destroy_node(self):
        pass


def reset():
    BUS.clear()
    LAST.clear()
    PARAMS.clear()
    SIM["t"] = 0.0


def install():
    """Register the fake modules in sys.modules (call before importing nodes)."""
    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    def cls(n):
        return type(n, (Msg,), {})

    rclpy = mod("rclpy", init=lambda args=None: None, shutdown=lambda: None,
                spin=lambda n: None, ok=lambda: True)
    rclpy.node = mod("rclpy.node", Node=Node)
    rclpy.qos = mod("rclpy.qos", QoSProfile=lambda **k: 10,
                    ReliabilityPolicy=types.SimpleNamespace(BEST_EFFORT=0, RELIABLE=1),
                    HistoryPolicy=types.SimpleNamespace(KEEP_LAST=0))
    eufs = ["WaypointArrayStamped", "Waypoint", "ConeArrayWithCovariance", "ConeWithCovariance", "CarState"]
    mod("eufs_msgs")
    mod("eufs_msgs.msg", **{n: cls(n) for n in eufs})
    mod("geometry_msgs")
    mod("geometry_msgs.msg", Point=cls("Point"), Twist=cls("Twist"), Pose=cls("Pose"))
    mod("visualization_msgs")
    mod("visualization_msgs.msg", Marker=Marker, MarkerArray=cls("MarkerArray"))
    mod("ackermann_msgs")
    mod("ackermann_msgs.msg", AckermannDriveStamped=cls("AckermannDriveStamped"))
    mod("nav_msgs")
    mod("nav_msgs.msg", Odometry=cls("Odometry"))
    mod("sensor_msgs")
    mod("sensor_msgs.msg", Imu=cls("Imu"), NavSatFix=cls("NavSatFix"))
    mod("std_srvs")
    mod("std_srvs.srv", Trigger=cls("Trigger"))
