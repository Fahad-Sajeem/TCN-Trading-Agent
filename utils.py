# -*- coding: utf-8 -*-
"""Utility functions"""

import datetime


def to_milliseconds(dt):
    """Convert datetime to milliseconds timestamp"""
    return int(dt.timestamp() * 1000)


def get_time_range(days=180):
    """Get start and end time in milliseconds for data fetching"""
    end_time = datetime.datetime.utcnow()
    start_time = end_time - datetime.timedelta(days=days)
    return to_milliseconds(start_time), to_milliseconds(end_time)

