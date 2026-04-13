#!/usr/bin/env python3
import csv
import math
import os
from typing import Optional

import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from rclpy.node import Node


class SteeringLoggerNode(Node):
    def __init__(self) -> None:
        super().__init__('steering_logger_node')

        self.declare_parameter('topic_name', '/moa/cmd_vel')
        self.declare_parameter('csv_path', './logs/steering_log.csv')
        self.declare_parameter('use_header_stamp', True)
        self.declare_parameter('print_every_n', 20)

        self.topic_name = self.get_parameter(
            'topic_name').get_parameter_value().string_value
        self.csv_path = self.get_parameter(
            'csv_path').get_parameter_value().string_value
        self.use_header_stamp = self.get_parameter(
            'use_header_stamp').get_parameter_value().bool_value
        self.print_every_n = max(1, self.get_parameter(
            'print_every_n').get_parameter_value().integer_value)

        self.sample_index = 0

        self.last_valid_time_sec: Optional[float] = None
        self.last_valid_theta: Optional[float] = None
        self.last_valid_dtheta: float = math.nan
        self.valid_derivative_sample_count = 0

        self.csv_file, self.csv_writer = self._open_csv(self.csv_path)

        self.subscription = self.create_subscription(
            AckermannDriveStamped,
            self.topic_name,
            self.cmd_vel_callback,
            10,
        )

        self.get_logger().info(
            f'Logging steering from {self.topic_name} to {self.csv_path} '
            f'(use_header_stamp={self.use_header_stamp}, print_every_n={self.print_every_n})'
        )

    def _open_csv(self, csv_path: str):
        directory = os.path.dirname(csv_path)
        if directory:
            os.makedirs(directory, exist_ok=True)

        file_exists = os.path.exists(csv_path)
        has_content = file_exists and os.path.getsize(csv_path) > 0

        try:
            csv_file = open(csv_path, 'a', newline='', encoding='utf-8')
        except OSError as exc:
            self.get_logger().error(
                f'Failed to open CSV file at {csv_path}: {exc}')
            raise

        csv_writer = csv.writer(csv_file)

        if not has_content:
            csv_writer.writerow([
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
            csv_file.flush()

        return csv_file, csv_writer

    @staticmethod
    def _stamp_to_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def cmd_vel_callback(self, msg: AckermannDriveStamped) -> None:
        recv_time_sec = self.get_clock().now().nanoseconds * 1e-9
        header_time_sec = self._stamp_to_sec(msg.header.stamp)
        msg_time_sec = header_time_sec if self.use_header_stamp else recv_time_sec

        theta = float(msg.drive.steering_angle)

        dt_sec = math.nan
        dtheta_dt = math.nan
        d2theta_dt2 = math.nan

        if self.last_valid_time_sec is None:
            self.last_valid_time_sec = msg_time_sec
            self.last_valid_theta = theta
            self.last_valid_dtheta = math.nan
            self.valid_derivative_sample_count = 1
        else:
            dt_sec = msg_time_sec - self.last_valid_time_sec
            if dt_sec > 0.0:
                dtheta_dt = (theta - self.last_valid_theta) / dt_sec
                if self.valid_derivative_sample_count >= 2 and math.isfinite(self.last_valid_dtheta):
                    d2theta_dt2 = (dtheta_dt - self.last_valid_dtheta) / dt_sec

                self.last_valid_time_sec = msg_time_sec
                self.last_valid_theta = theta
                self.last_valid_dtheta = dtheta_dt
                self.valid_derivative_sample_count += 1

        self.csv_writer.writerow([
            self.sample_index,
            recv_time_sec,
            msg_time_sec,
            dt_sec,
            theta,
            float(msg.drive.steering_angle_velocity),
            float(msg.drive.speed),
            float(msg.drive.acceleration),
            float(msg.drive.jerk),
            dtheta_dt,
            d2theta_dt2,
        ])
        self.csv_file.flush()

        self.sample_index += 1

        if self.sample_index % self.print_every_n == 0:
            self.get_logger().info(
                f'samples={self.sample_index}, theta={theta:.6f}, dt={dt_sec}, '
                f'dtheta_dt={dtheta_dt}, d2theta_dt2={d2theta_dt2}'
            )

    def destroy_node(self) -> bool:
        try:
            if hasattr(self, 'csv_file') and self.csv_file and not self.csv_file.closed:
                self.csv_file.flush()
                self.csv_file.close()
        finally:
            return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SteeringLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('KeyboardInterrupt received, shutting down steering_logger_node.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
