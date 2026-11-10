"""Tınaztepe Akıllı Rotalama Ajanı - MDP modeli ve ajan iskeleti.

Formülasyon (Stanford CS221 tarzı, Stokastik En Kısa Yol problemi):
    Durum   s = (düğüm, saat[dk], ulaşım türü)
    Eylem   a = (tür, hedef düğüm, kenar anahtarı)
    Geçiş   T(s' | s, a): trafik sıkışıklığı veya sefer rötarı sonucu varış
            süresinin olasılık dağılımı (iki sonuçlu: normal / gecikmeli).
    Maliyet Cost(s, a, s') = cost_functions içindeki dakika maliyeti
    Hedef   Tınaztepe düğümüne ulaşmak (terminal, maliyet 0)

Bellman denklemi (minimizasyon, iskonto gamma):
    Q(s, a) = sum_{s'} T(s'|s,a) * [Cost(s,a,s') + gamma * V(s')]
    V(s)    = min_a Q(s, a)
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Hashable, List, Optional, Tuple

import networkx as nx

try:  # paket olarak veya doğrudan çalıştırma
    from core.cost_functions import (
        MAX_TRAFFIC_MULT, MIN_TRAFFIC_MULT, calculate_edge_cost,
    )
except ImportError:  # pragma: no cover
    from cost_functions import (
        MAX_TRAFFIC_MULT, MIN_TRAFFIC_MULT, calculate_edge_cost,
    )

TIME_BUCKET_MIN = 5  # Value Iteration için zaman ayrıklaştırma adımı (dk)


class ActionType(Enum):
    """Eylem türleri."""
    DRIVE = "drive"
    BOARD = "board"        # hatta binme
    TRANSFER = "transfer"  # aktarma
    WALK = "walk"


@dataclass(frozen=True)
class TinaztepeMDPState:
    """MDP durumu: mevcut düğüm/durak, günün saati (dakika, 0-1439+), ulaşım türü."""
    node: Hashable
    time_min: float
    mode: str  # "car" | "transit"

    def key(self) -> Tuple[Hashable, int, str]:
        """Zamanı TIME_BUCKET_MIN adımına yuvarlanmış, hashlenebilir anahtar."""
        bucket = int(round(self.time_min / TIME_BUCKET_MIN))
        return (self.node, bucket, self.mode)


@dataclass(frozen=True)
class TinaztepeAction:
    """MDP eylemi: komşu düğüme sürüş / binme / aktarma / yürüme."""
    action_type: ActionType
    target: Hashable
    edge_key: int = 0


# traffic_fn(saat_dk, kenar_verisi) -> trafik katsayısı (1.0 - 2.5)
TrafficFn = Callable[[float, dict], float]
# delay_fn(saat_dk, kenar_verisi) -> sefer rötar olasılığı (0-1)
DelayFn = Callable[[float, dict], float]


def default_traffic_multiplier(time_min: float, edge: dict) -> float:
    """Yer tutucu: sabah (07:30-09:30) ve akşam (17:00-19:00) yoğunluğu.

    K1'in gün/saat bazlı gecikme matrisi hazır olunca değiştirilecektir.
    """
    h = (time_min / 60.0) % 24
    if 7.5 <= h <= 9.5 or 17 <= h <= 19:
        return 2.0
    if 6.5 <= h < 7.5 or 9.5 < h <= 10.5 or 16 <= h < 17:
        return 1.4
    return 1.0


def default_delay_probability(time_min: float, edge: dict) -> float:
    """Yer tutucu: kenardaki 'delay_risk' (varsayılan 0.15), yoğun saatte artar."""
    base = edge.get("delay_risk", 0.15)
    return min(0.9, base * (1.5 if default_traffic_multiplier(time_min, edge) > 1.5 else 1.0))


class RoutingMDPAgent:
    """Belirsizlik altında beklenen süreyi minimize eden MDP rotalama ajanı.

    Args:
        graph: Kenarları cost_functions sözleşmesine uygun NetworkX (Multi)DiGraph.
        goal_node: Hedef (Tınaztepe) düğümü.
        traffic_fn: Zamana bağlı trafik katsayısı fonksiyonu.
        delay_fn: Zamana bağlı sefer rötar olasılığı fonksiyonu.
        discount: İskonto faktörü gamma (0-1]; 1.0 = standart stokastik en kısa yol.
        transit_delay_factor: Rötarda bekleme süresi çarpanı.
    """

    def __init__(
        self,
        graph: nx.Graph,
        goal_node: Hashable,
        traffic_fn: TrafficFn = default_traffic_multiplier,
        delay_fn: DelayFn = default_delay_probability,
        discount: float = 1.0,
        transit_delay_factor: float = 2.5,
    ) -> None:
        self.graph = graph
        self.goal_node = goal_node
        self.traffic_fn = traffic_fn
        self.delay_fn = delay_fn
        self.discount = discount
        self.transit_delay_factor = transit_delay_factor
        self.values: Dict[Tuple, float] = {}

    # ------------------------------------------------------------------ MDP
    def is_end(self, state: TinaztepeMDPState) -> bool:
        """Hedefe ulaşıldı mı (terminal durum)."""
        return state.node == self.goal_node

    def _edge_iter(self, node: Hashable):
        """Düğümden çıkan (hedef, anahtar, veri) üçlüleri; Multi ve düz graf uyumlu."""
        if self.graph.is_multigraph():
            for _, v, k, d in self.graph.out_edges(node, keys=True, data=True):
                yield v, k, d
        else:
            for _, v, d in self.graph.out_edges(node, data=True):
                yield v, 0, d

    def _edge_data(self, node: Hashable, action: TinaztepeAction) -> dict:
        if self.graph.is_multigraph():
            return self.graph[node][action.target][action.edge_key]
        return self.graph[node][action.target]

    @staticmethod
    def _action_type_for(edge: dict) -> ActionType:
        mode = edge.get("mode", "drive")
        if mode == "transfer" or edge.get("transfer", False):
            return ActionType.TRANSFER
        return {"drive": ActionType.DRIVE, "walk": ActionType.WALK,
                "transit": ActionType.BOARD}.get(mode, ActionType.DRIVE)

    def get_actions(self, state: TinaztepeMDPState) -> List[TinaztepeAction]:
        """Durumdaki ulaşım türüne uygun komşu düğüm eylemlerini döndürür.

        Araba: drive kenarları. Toplu taşıma: walk/transit/transfer kenarları.
        """
        if self.is_end(state):
            return []
        allowed = {"drive"} if state.mode == "car" else {"walk", "transit", "transfer"}
        return [
            TinaztepeAction(self._action_type_for(d), v, k)
            for v, k, d in self._edge_iter(state.node)
            if d.get("mode", "drive") in allowed
        ]

    def get_transition_states_and_probs(
        self, state: TinaztepeMDPState, action: TinaztepeAction
    ) -> List[Tuple[TinaztepeMDPState, float, float]]:
        """T(s'|s,a): (sonraki durum, olasılık, maliyet[dk]) üçlüleri listesi.

        - DRIVE: trafik katsayısı m için hafif (0.85m) / ağır (1.4m) sıkışıklık;
          ağır sıkışıklık olasılığı p = (m-1)/1.5 (0.05-0.8 aralığında).
        - BOARD/TRANSFER: sefer zamanında (p=1-r) veya rötarlı (p=r), rötarda
          bekleme süresi transit_delay_factor kat artar.
        - WALK: deterministik.
        """
        edge = self._edge_data(state.node, action)
        t = state.time_min
        outcomes: List[Tuple[float, float]] = []  # (olasılık, maliyet)

        if action.action_type == ActionType.DRIVE:
            m = self.traffic_fn(t, edge)
            light = max(MIN_TRAFFIC_MULT, m * 0.85)
            heavy = min(MAX_TRAFFIC_MULT, m * 1.4)
            p_heavy = min(0.8, max(0.05, (m - 1.0) / 1.5))
            outcomes = [
                (1 - p_heavy, calculate_edge_cost(edge, traffic_multiplier=light)),
                (p_heavy, calculate_edge_cost(edge, traffic_multiplier=heavy)),
            ]
        elif action.action_type in (ActionType.BOARD, ActionType.TRANSFER):
            r = self.delay_fn(t, edge)
            outcomes = [
                (1 - r, calculate_edge_cost(edge, delay_risk_multiplier=1.0)),
                (r, calculate_edge_cost(edge, delay_risk_multiplier=self.transit_delay_factor)),
            ]
        else:
            outcomes = [(1.0, calculate_edge_cost(edge))]

        return [
            (TinaztepeMDPState(action.target, t + cost, state.mode), p, cost)
            for p, cost in outcomes if p > 0
        ]

    def get_cost(self, state: TinaztepeMDPState, action: TinaztepeAction,
                 next_state: TinaztepeMDPState) -> float:
        """Cost(s,a,s'): geçişte harcanan süre (dk) = varış - çıkış zamanı."""
        return next_state.time_min - state.time_min

    # ------------------------------------------------------- Bellman / VI
    def compute_q_value(self, state: TinaztepeMDPState, action: TinaztepeAction) -> float:
        """Q(s,a) = sum_{s'} T * [Cost + gamma * V(s')]; V bilinmiyorsa 0 alınır."""
        q = 0.0
        for nxt, p, cost in self.get_transition_states_and_probs(state, action):
            q += p * (cost + self.discount * self.get_value(nxt))
        return q

    def get_value(self, state: TinaztepeMDPState) -> float:
        """V(s): terminalde 0, bilinmeyen durumda 0 (iyimser başlangıç)."""
        if self.is_end(state):
            return 0.0
        return self.values.get(state.key(), 0.0)

    def compute_value(self, state: TinaztepeMDPState) -> Tuple[float, Optional[TinaztepeAction]]:
        """V(s) = min_a Q(s,a) ve en iyi eylem (argmin)."""
        actions = self.get_actions(state)
        if not actions:
            return 0.0, None
        qs = [(self.compute_q_value(state, a), a) for a in actions]
        return min(qs, key=lambda x: x[0])

    def value_iteration(self, start: TinaztepeMDPState, max_iters: int = 50,
                        epsilon: float = 1e-3, horizon_min: float = 240.0) -> Dict[Tuple, float]:
        """Başlangıçtan erişilebilir durumlar (zaman ufku içinde) üzerinde Value Iteration.

        Zaman ayrıklaştırıldığı için durum uzayı sonludur; yakınsayınca durur.
        """
        states = self._reachable_states(start, horizon_min)
        for _ in range(max_iters):
            delta = 0.0
            for s in states:
                if self.is_end(s):
                    continue
                v_new, _ = self.compute_value(s)
                delta = max(delta, abs(v_new - self.values.get(s.key(), 0.0)))
                self.values[s.key()] = v_new
            if delta < epsilon:
                break
        return self.values

    def _reachable_states(self, start: TinaztepeMDPState, horizon_min: float):
        seen = {start.key(): start}
        stack = [start]
        while stack:
            s = stack.pop()
            for a in self.get_actions(s):
                for nxt, _, _ in self.get_transition_states_and_probs(s, a):
                    if nxt.time_min - start.time_min > horizon_min:
                        continue
                    if nxt.key() not in seen:
                        seen[nxt.key()] = nxt
                        stack.append(nxt)
        return list(seen.values())

    def extract_policy(self, start: TinaztepeMDPState, max_steps: int = 100) -> List[TinaztepeAction]:
        """Value Iteration sonrası açgözlü politika ile beklenen rotayı çıkarır
        (her adımda en olası sonuç izlenir)."""
        path, s = [], start
        for _ in range(max_steps):
            if self.is_end(s):
                break
            _, a = self.compute_value(s)
            if a is None:
                break
            path.append(a)
            s = max(self.get_transition_states_and_probs(s, a), key=lambda x: x[1])[0]
        return path


if __name__ == "__main__":
    # Mock graf: Evim(A) -> B -> Tınaztepe(T); araba ve toplu taşıma kenarları
    g = nx.MultiDiGraph()
    g.add_edge("A", "B", mode="drive", length_km=4, speed_kph=50, signal_wait_min=1.0)
    g.add_edge("B", "T", mode="drive", length_km=6, speed_kph=70, signal_wait_min=0.5)
    g.add_edge("A", "S1", mode="walk", walk_min=5)
    g.add_edge("S1", "S2", mode="transit", wait_min=6, in_vehicle_min=18, delay_risk=0.2)
    g.add_edge("S2", "T", mode="transfer", walk_min=3, wait_min=5, in_vehicle_min=10,
               delay_risk=0.3)

    agent = RoutingMDPAgent(g, goal_node="T")
    for label, mode, minute in [("Araba (08:00, yoğun)", "car", 480),
                                ("Araba (13:00, sakin)", "car", 780),
                                ("Toplu taşıma (08:00)", "transit", 480)]:
        start = TinaztepeMDPState("A", minute, mode)
        agent.values.clear()
        agent.value_iteration(start)
        v, best = agent.compute_value(start)
        route = [a.target for a in agent.extract_policy(start)]
        print(f"{label}: beklenen süre = {v:.1f} dk, rota = A -> {' -> '.join(map(str, route))}")

    s0 = TinaztepeMDPState("A", 480, "car")
    a0 = agent.get_actions(s0)[0]
    print("Geçişler (s', olasılık, maliyet):")
    for ns, p, c in agent.get_transition_states_and_probs(s0, a0):
        print(f"  -> {ns.node} @ {ns.time_min:.1f} dk, p={p:.2f}, maliyet={c:.1f} dk")

