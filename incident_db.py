from __future__ import annotations

import sqlite3
from typing import Dict


DB_PATH = "incident_logs.db"


def init_database(db_path: str = DB_PATH) -> None:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS incident_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                timestamp_sec REAL NOT NULL,
                temporal_probability REAL NOT NULL,
                image_path TEXT NOT NULL,
                map_path TEXT,
                camera_id TEXT NOT NULL,
                location TEXT NOT NULL,
                latitude REAL NOT NULL,
                longitude REAL NOT NULL,
                vehicles TEXT,
                fire_regions INTEGER NOT NULL,
                plates TEXT,
                nearest_hospital TEXT,
                nearest_police_station TEXT,
                legacy_classifier TEXT,
                whatsapp_number TEXT
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def insert_incident_log(record: Dict[str, object], db_path: str = DB_PATH) -> None:
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()
        cursor.execute(
            """
            INSERT INTO incident_logs (
                timestamp_sec,
                temporal_probability,
                image_path,
                map_path,
                camera_id,
                location,
                latitude,
                longitude,
                vehicles,
                fire_regions,
                plates,
                nearest_hospital,
                nearest_police_station,
                legacy_classifier,
                whatsapp_number
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["timestamp_sec"],
                record["temporal_probability"],
                record["image_path"],
                record.get("map_path"),
                record["camera_id"],
                record["location"],
                record["latitude"],
                record["longitude"],
                record["vehicles"],
                record["fire_regions"],
                record["plates"],
                record["nearest_hospital"],
                record["nearest_police_station"],
                record["legacy_classifier"],
                record["whatsapp_number"],
            ),
        )
        connection.commit()
    finally:
        connection.close()
