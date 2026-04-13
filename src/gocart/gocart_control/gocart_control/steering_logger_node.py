#!/usr/bin/env python3
import csv
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped


class SteeringLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__('steering_logger_node')

        self.declare_parameter('topic_name', '/moa/cmd_vel')
        self.declare_parameter('csv_path', './logs/steering_log.csv')
        self.declare_parameter('use_header_stamp', True)
        self.declare_parameter('print_every_n', 20)

        self.topic_name = self.get_parameter('topic_name').get_parameter_value().string_value
        self.csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.use_header_stamp = self.get_parameter('use_header_stamp').get_parameter_value().bool_value
        self.print_every_n = self.get_parameter('print_every_n').get_parameter_value().integer_value

        if self.print_every_n <= 0:
            self.get_logger().warn('print_every_n must be > 0. Falling back to 20.')
            self.print_every_n = 20

        self.sample_index = 0

        self.prev_time_sec = None
        self.prev_theta = None
        self.prev_dtheta_dt = None

        self._open_csv()

        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            self.topic_name,
            self._callback,
            10,
        )

        self.get_logger().info(
            f'Steering logger started. topic={self.topic_name}, csv_path={self.csv_path}, '
            f'use_header_stamp={self.use_header_stamp}, print_every_n={self.print_every_n}'
        )

    def _open_csv(self) -> None:
        csv_file = Path(self.csv_path)
        csv_file.parent.mkdir(parents=True, exist_ok=True)

        file_exists = csv_file.exists()
        self.csv_handle = open(csv_file, 'a', newline='', encoding='utf-8')
        self.csv_writer = csv.writer(self.csv_handle)

        if (not file_exists) or csv_file.stat().st_size == 0:
            self.csv_writer.writerow([
                'sample_index',
                'recv_time_sec',
                'msg_time_sec',
                'dt_sec',
                'steering_angle',
                'steering_angle_velocity_msg',
                'speed',
                'acceleration',
                'jerk',
                'dtheta_dt',
                'd2theta_dt2',
            ])
            self.csv_handle.flush()

    def _stamp_to_sec(self, msg: AckermannDriveStamped) -> float:
        return float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9

    def _callback(self, msg: AckermannDriveStamped) -> None:
        recv_time_sec = self.get_clock().now().nanoseconds * 1e-9
        header_time_sec = self._stamp_to_sec(msg)

        if self.use_header_stamp:
            msg_time_sec = header_time_sec
        else:
            msg_time_sec = recv_time_sec

        steering_angle = float(msg.drive.steering_angle)
        steering_angle_velocity_msg = float(msg.drive.steering_angle_velocity)
        speed = float(msg.drive.speed)
        acceleration = float(msg.drive.acceleration)
        jerk = float(msg.drive.jerk)

        dt_sec = math.nan
        dtheta_dt = math.nan
        d2theta_dt2 = math.nan

        if self.prev_time_sec is None:
            self.prev_time_sec = msg_time_sec
            self.prev_theta = steering_angle
        else:
            dt_sec = msg_time_sec - self.prev_time_sec

            if dt_sec > 0.0:
                dtheta_dt = (steering_angle - self.prev_theta) / dt_sec

                if self.prev_dtheta_dt is not None:
                    d2theta_dt2 = (dtheta_dt - self.prev_dtheta_dt) / dt_sec

                self.prev_time_sec = msg_time_sec
                self.prev_theta = steering_angle
                self.prev_dtheta_dt = dtheta_dt

        self.csv_writer.writerow([
            self.sample_index,
            recv_time_sec,
            msg_time_sec,
            dt_sec,
            steering_angle,
            steering_angle_velocity_msg,
            speed,
            acceleration,
            jerk,
            dtheta_dt,
            d2theta_dt2,
        ])
        self.csv_handle.flush()

        self.sample_index += 1

        if self.sample_index % self.print_every_n == 0:
            self.get_logger().info(
                f'logged {self.sample_index} samples. '
                f'latest theta={steering_angle:.6f}, dtheta_dt={dtheta_dt}, d2theta_dt2={d2theta_dt2}'
            )

    def close(self) -> None:
        if hasattr(self, 'csv_handle') and self.csv_handle:
            self.csv_handle.flush()
            self.csv_handle.close()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SteeringLoggerNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received. Shutting down steering_logger_node.')
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
