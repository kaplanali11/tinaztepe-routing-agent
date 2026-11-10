"""Tınaztepe Akıllı Rotalama Ajanı - Maliyet fonksiyonları.

Tüm maliyetler DAKİKA cinsindendir (negatif fayda = beklenen seyahat süresi).
Araba ve toplu taşıma için ayrı fonksiyonlar ile kenar tipine göre otomatik
seçim yapan genel bir fonksiyon sağlar.

Kenar (edge) öznitelik sözleşmesi (K1 ile ortak):
    mode            : "drive" | "walk" | "transit" | "transfer"
    length_km       : segment uzunluğu (km)
    speed_kph       : hız sınırı (km/s)
    signal_wait_min : kavşak/ışık bekleme süresi (dk)
    walk_min        : yürüme süresi (dk)
    wait_min        : durakta nominal bekleme (dk, ~headway/2)
    in_vehicle_min  : araç içi süre (dk)
    transfer        : aktarma var mı (bool)
"""
from __future__ import annotations

from typing import Any, Mapping

MIN_TRAFFIC_MULT = 1.0
MAX_TRAFFIC_MULT = 2.5
DEFAULT_TRANSFER_PENALTY_MIN = 5.0
DEFAULT_WALK_SPEED_KPH = 4.5


def clamp_traffic_multiplier(multiplier: float) -> float:
    """Trafik katsayısını [1.0, 2.5] aralığına sınırlar."""
    return max(MIN_TRAFFIC_MULT, min(MAX_TRAFFIC_MULT, multiplier))


def calculate_car_cost(
    length_km: float,
    speed_kph: float,
    traffic_multiplier: float = 1.0,
    signal_wait_min: float = 0.0,
) -> float:
    """Bir yol segmenti için beklenen araç süresini (dk) hesaplar.

    cost = (length / speed) * 60 * traffic_multiplier + signal_wait

    Args:
        length_km: Segment uzunluğu (km).
        speed_kph: Hız sınırı (km/s), > 0 olmalı.
        traffic_multiplier: Gün/saat trafik katsayısı (1.0 - 2.5).
        signal_wait_min: Kavşak/ışık ortalama bekleme süresi (dk).
    """
    if speed_kph <= 0:
        raise ValueError("speed_kph pozitif olmalıdır.")
    if length_km < 0 or signal_wait_min < 0:
        raise ValueError("length_km ve signal_wait_min negatif olamaz.")
    mult = clamp_traffic_multiplier(traffic_multiplier)
    return (length_km / speed_kph) * 60.0 * mult + signal_wait_min


def calculate_transit_cost(
    walk_min: float,
    wait_min: float,
    in_vehicle_min: float,
    delay_risk_multiplier: float = 1.0,
    is_transfer: bool = False,
    transfer_penalty_min: float = DEFAULT_TRANSFER_PENALTY_MIN,
) -> float:
    """Toplu taşıma segmenti için toplam maliyeti (negatif fayda, dk) hesaplar.

    cost = walk + wait * delay_risk_multiplier + in_vehicle
           (+ transfer_penalty, aktarma varsa)

    Args:
        walk_min: Durağa/duraktan yürüme süresi (dk).
        wait_min: Durakta nominal bekleme süresi (dk).
        in_vehicle_min: Araç içi yolculuk süresi (dk).
        delay_risk_multiplier: Sefer gecikme riski çarpanı (>= 1.0); rötarlı
            durumda bekleme süresini büyütür.
        is_transfer: Aktarma yapılıyor mu.
        transfer_penalty_min: Aktarma cezası (dk).
    """
    if min(walk_min, wait_min, in_vehicle_min) < 0:
        raise ValueError("Süreler negatif olamaz.")
    if delay_risk_multiplier < 1.0:
        raise ValueError("delay_risk_multiplier >= 1.0 olmalıdır.")
    cost = walk_min + wait_min * delay_risk_multiplier + in_vehicle_min
    if is_transfer:
        cost += transfer_penalty_min
    return cost


def calculate_edge_cost(
    edge_data: Mapping[str, Any],
    traffic_multiplier: float = 1.0,
    delay_risk_multiplier: float = 1.0,
) -> float:
    """Graf kenarının tipine (mode) göre maliyetini (dk) otomatik döndürür.

    Args:
        edge_data: Kenar öznitelik sözlüğü (modül başındaki sözleşmeye bakın).
        traffic_multiplier: Araba kenarları için trafik katsayısı.
        delay_risk_multiplier: Toplu taşıma kenarları için gecikme çarpanı.
    """
    mode = edge_data.get("mode", "drive")
    if mode == "drive":
        return calculate_car_cost(
            edge_data["length_km"],
            edge_data.get("speed_kph", 50.0),
            traffic_multiplier,
            edge_data.get("signal_wait_min", 0.0),
        )
    if mode == "walk":
        if "walk_min" in edge_data:
            return float(edge_data["walk_min"])
        return edge_data["length_km"] / DEFAULT_WALK_SPEED_KPH * 60.0
    if mode in ("transit", "transfer"):
        return calculate_transit_cost(
            edge_data.get("walk_min", 0.0),
            edge_data.get("wait_min", 0.0),
            edge_data.get("in_vehicle_min", 0.0),
            delay_risk_multiplier,
            is_transfer=(mode == "transfer" or edge_data.get("transfer", False)),
            transfer_penalty_min=edge_data.get(
                "transfer_penalty_min", DEFAULT_TRANSFER_PENALTY_MIN
            ),
        )
    raise ValueError(f"Bilinmeyen kenar tipi: {mode}")


if __name__ == "__main__":
    print("Araba   :", round(calculate_car_cost(3.0, 50, 1.8, 0.5), 2), "dk")
    print("Otobüs  :", round(calculate_transit_cost(4, 6, 20, 1.5, True), 2), "dk")
    print("Kenar   :", round(calculate_edge_cost(
        {"mode": "drive", "length_km": 2, "speed_kph": 60}, 1.5), 2), "dk")

