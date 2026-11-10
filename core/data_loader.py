import osmnx as ox
import pandas as pd
import os

# Yol ağı ve verilerin tutulacağı klasör yolları
GRAPH_PATH = "data/izmir_drive_graph.graphml"
GTFS_DIR = "data/gtfs_buca/"

def load_or_create_graph():
    """
    Buca sürüş yol ağını OpenStreetMap üzerinden çeker.
    Her seferinde baştan indirmemek için grafı yerel dosyaya kaydeder.
    """
    if os.path.exists(GRAPH_PATH):
        print("✅ Harita yerel dosyadan yüklüyor...")
        G = ox.load_graphml(GRAPH_PATH)
    else:
        print("🌐 Harita OpenStreetMap üzerinden çekiliyor, biraz sürebilir...")
        # Buca bölgesinin sürüş ağını indir
        G = ox.graph_from_place("Buca, Izmir, Turkey", network_type='drive')
        
        # Grafa hız ve seyahat süresi parametrelerini hesaplatıp ekle
        G = ox.add_edge_speeds(G)
        G = ox.add_edge_travel_times(G)
        
        # Klasör yoksa açıp grafı kaydet
        os.makedirs("data", exist_ok=True)
        ox.save_graphml(G, GRAPH_PATH)
        print("💾 Harita grafı 'data/' klasörüne kaydedildi!")
        
    return G

def load_gtfs_stops():
    """
    ESHOT ve İZBAN durak verilerini (stops.txt) okur.
    """
    stops_file = os.path.join(GTFS_DIR, "stops.txt")
    if os.path.exists(stops_file):
        print("✅ GTFS durak verileri yükleniyor...")
        return pd.read_csv(stops_file)
    else:
        print("⚠️ Uyarı: GTFS stops.txt dosyası bulunamadı.")
        return pd.DataFrame(columns=["stop_id", "stop_name", "stop_lat", "stop_lon"])

if __name__ == "__main__":
    print("--- Yol Haritası Veri Yükleme Testi ---")
    G = load_or_create_graph()
    print(f"Toplam Düğüm (Kavşak) Sayısı: {len(G.nodes)}")
    print(f"Toplam Kenar (Sokak) Sayısı: {len(G.edges)}")
    print("--- İşlem Başarıyla Tamamlandı ---")
    # GTFS Durak testi
    stops = load_gtfs_stops()
    print(f"Yüklenen Durak Sayısı: {len(stops)}")
    print("--- İşlem Başarıyla Tamamlandı ---")