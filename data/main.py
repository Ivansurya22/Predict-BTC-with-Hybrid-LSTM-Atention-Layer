import sqlite3
import os
import pandas as pd

# ==============================
# KONFIGURASI
# ==============================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
db_path = os.path.join(BASE_DIR, "btc_ohlcv.db")

print(f"📂 Membuka: {db_path}")

# ==============================
# KONEKSI DATABASE
# ==============================
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Ambil semua tabel
cursor.execute(
    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%';"
)
tables = [t[0] for t in cursor.fetchall()]

# ==============================
# CETAK STRUKTUR DAN DATA
# ==============================
if not tables:
    print("⚠️ Tidak ada tabel ditemukan.")
else:
    for table in tables:
        print(f"\n🧱 Tabel: {table}")

        # Ambil info kolom
        cursor.execute(f"PRAGMA table_info({table});")
        columns = cursor.fetchall()
        if columns:
            print("  📑 Kolom:")
            for col in columns:
                print(f"   - {col[1]} ({col[2]})")
        else:
            print("  ⚠️ Tidak ada kolom ditemukan.")

        # Ambil 5 data pertama (HEAD)
        query = f"SELECT * FROM {table} LIMIT 5;"
        df = pd.read_sql_query(query, conn)

        if not df.empty:
            print("\n  📊 5 Data Pertama:")
            print(df.to_string(index=False))
        else:
            print("  ⚠️ Tabel kosong (tidak ada data).")

conn.close()
