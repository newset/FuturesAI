"""
strategies/ema20.py — EMA20 移动平均线策略

基于20周期EMA的顺势交易策略：
  1. 价格上穿EMA20且K线收盘 > EMA20 → 多头信号
  2. 价格下穿EMA20且K线收盘 < EMA20 → 空头信号
  3. 配合K线强度评级（强/中/弱）过滤虚假信号
  4. EMA缺口分析（连续不触及EMA的K线数）判断趋势强度
"""

from __future__ import annotations
import pandas as pd
from strategies.base import BaseStrategy

class EMA20Strategy(BaseStrategy):
    """EMA20 移动平均线顺势交易策略"""

    # ── 标识 ──────────────────────────────────────────────────────────────────

    @property
    def strategy_id(self) -> str:
        """唯一标识符"""
        return "ema20"

    # ── 默认值 ────────────────────────────────────────────────────────────────

    _PRICE_STRUCTURE_DEFAULTS = {
        "trend_direction": "neutral",  # "up" / "down" / "neutral"
        "ema20_position": "unknown",  # "above" / "below" / "near"
        "ema20_gap_bars": 0,  # EMA缺口K线数
        "ema20_distance_pct": 0.0,  # 距EMA幅度百分比
        "last_signal_bar_rating": "unknown",  # 最后信号棒评级
        "ema20": {}  # EMA20详细数据
    }

    @property
    def price_structure_defaults(self) -> dict:
        """策略预计算字段默认值"""
        return self._PRICE_STRUCTURE_DEFAULTS

    # ── 价格结构计算 ───────────────────────────────────────────────────────────

    def build_price_structure(self, df: pd.DataFrame) -> dict:
        """计算EMA20及相关指标"""
        result = dict(self._PRICE_STRUCTURE_DEFAULTS)
        
        if len(df) < 20:
            result["ema20"] = {}
            return result
        
        # 计算EMA20
        df = df.copy()
        df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
        
        # 提取最后几根K线的数据
        current_bar = df.iloc[-1]
        prev_bar = df.iloc[-2] if len(df) >= 2 else None
        
        ema20_value = current_bar["ema20"]
        close_price = current_bar["close"]
        
        # 判断价格相对EMA20的位置
        distance_pct = ((close_price - ema20_value) / ema20_value) * 100
        
        if abs(distance_pct) <= 0.5:
            ema_position = "near"
        elif distance_pct > 0:
            ema_position = "above"
        else:
            ema_position = "below"
        
        # 判断趋势方向
        if distance_pct > 0:
            trend = "up"
        elif distance_pct < 0:
            trend = "down"
        else:
            trend = "neutral"
        
        # 计算EMA缺口K线数（连续不触及EMA的K线数）
        gap_bars = 0
        for i in range(len(df) - 1, max(0, len(df) - 50), -1):
            bar = df.iloc[i]
            # 如果K线低点 >= EMA20或高点 <= EMA20，则未触及EMA
            if bar["high"] < ema20_value or bar["low"] > ema20_value:
                gap_bars += 1
            else:
                break
        
        # 构建EMA20详细信息字典
        ema20_detail = {
            "value": round(ema20_value, 2),
            "distance_pct": round(distance_pct, 2),
            "position": ema_position,
            "gap_bars": gap_bars,
            "close": round(close_price, 2),
            "high": round(current_bar["high"], 2),
            "low": round(current_bar["low"], 2),
            "ema_above_close": close_price < ema20_value,
        }
        
        # 添加前一根K线的EMA穿越信息
        if prev_bar is not None and "ema20" in df.columns:
            prev_ema20 = df.iloc[-2]["ema20"]
            prev_close = prev_bar["close"]
            
            # 判断是否穿越EMA
            crossed_above = (prev_close <= prev_ema20 and close_price > ema20_value)
            crossed_below = (prev_close >= prev_ema20 and close_price < ema20_value)
            
            ema20_detail["crossed_above"] = crossed_above
            ema20_detail["crossed_below"] = crossed_below
            ema20_detail["prev_close"] = round(prev_close, 2)
            ema20_detail["prev_ema20"] = round(prev_ema20, 2)
        
        result["ema20"] = ema20_detail
        result["trend_direction"] = trend
        result["ema20_position"] = ema_position
        result["ema20_gap_bars"] = gap_bars
        result["ema20_distance_pct"] = distance_pct
        
        return result

    # ── 特征块构建 ─────────────────────────────────────────────────────────────

    def build_features(
        self,
        market_data: dict,
        lang: str,
        klines_completed: list[dict] | None = None,
    ) -> dict:
        """构建注入AI prompt的特征块"""
        
        ema20_info = market_data.get("ema20", {})
        quote = market_data.get("quote", {})
        
        # 中文特征
        if lang == "zh":
            features = {
                "EMA20分析": {
                    "EMA20价值": ema20_info.get("value", "未知"),
                    "价格位置": self._get_position_label_zh(ema20_info.get("position")),
                    "距离EMA幅度%": ema20_info.get("distance_pct", 0),
                    "EMA缺口K线数": ema20_info.get("gap_bars", 0),
                    "趋势方向": self._get_trend_label_zh(market_data.get("trend_direction")),
                },
                "K线信息": {
                    "收盘价": ema20_info.get("close", "未知"),
                    "最高价": ema20_info.get("high", "未知"),
                    "最低价": ema20_info.get("low", "未知"),
                    "EMA在收盘价上方": ema20_info.get("ema_above_close", False),
                },
                "穿越信号": {
                    "上穿EMA": ema20_info.get("crossed_above", False),
                    "下穿EMA": ema20_info.get("crossed_below", False),
                },
            }
        else:
            # 英文特征
            features = {
                "EMA20_Analysis": {
                    "EMA20_Value": ema20_info.get("value", "Unknown"),
                    "Price_Position": self._get_position_label_en(ema20_info.get("position")),
                    "Distance_From_EMA_Pct": ema20_info.get("distance_pct", 0),
                    "EMA_Gap_Bars": ema20_info.get("gap_bars", 0),
                    "Trend_Direction": self._get_trend_label_en(market_data.get("trend_direction")),
                },
                "Bar_Information": {
                    "Close_Price": ema20_info.get("close", "Unknown"),
                    "High": ema20_info.get("high", "Unknown"),
                    "Low": ema20_info.get("low", "Unknown"),
                    "EMA_Above_Close": ema20_info.get("ema_above_close", False),
                },
                "Crossover_Signals": {
                    "Crossed_Above_EMA": ema20_info.get("crossed_above", False),
                    "Crossed_Below_EMA": ema20_info.get("crossed_below", False),
                },
            }
        
        # 评级最后一根K线
        features["_last_bar_rating"] = "待评级"
        
        return features

    # ── 辅助方法 ───────────────────────────────────────────────────────────────

    def _get_position_label_zh(self, position: str | None) -> str:
        """中文位置标签"""
        mapping = {
            "above": "在EMA20上方",
            "below": "在EMA20下方",
            "near": "在EMA20附近",
        }
        return mapping.get(position, "未知")

    def _get_position_label_en(self, position: str | None) -> str:
        """英文位置标签"""
        mapping = {
            "above": "Above_EMA20",
            "below": "Below_EMA20",
            "near": "Near_EMA20",
        }
        return mapping.get(position, "Unknown")

    def _get_trend_label_zh(self, trend: str | None) -> str:
        """中文趋势标签"""
        mapping = {
            "up": "上升趋势",
            "down": "下降趋势",
            "neutral": "中性",
        }
        return mapping.get(trend, "未知")

    def _get_trend_label_en(self, trend: str | None) -> str:
        """英文趋势标签"""
        mapping = {
            "up": "Uptrend",
            "down": "Downtrend",
            "neutral": "Neutral",
        }
        return mapping.get(trend, "Unknown")

    # ── 跳过集合 ───────────────────────────────────────────────────────────────

    def get_skip_struct(self) -> frozenset[str]:
        """已在特征块中的字段，从原始market_data中剥离"""
        return frozenset([
            "trend_direction",
            "ema20_position",
            "ema20_gap_bars",
            "ema20_distance_pct",
            "last_signal_bar_rating",
            "ema20",
        ])

    # ── 信号棒评级 ─────────────────────────────────────────────────────────────

    def rate_bar(self, bar: dict, prev_bar: dict | None = None) -> str:
        """
        对K线进行强度评级。
        
        评级规则：
        - 多头棒：收盘 > 开盘
        - 空头棒：收盘 < 开盘
        - 十字星：收盘 ≈ 开盘
        
        强度基于：
        1. 实体大小（close_ratio）
        2. 上下影线（high_low_ratio）
        3. 方向性（是否超过前一根）
        """
        open_price = bar.get("open", 0)
        close_price = bar.get("close", 0)
        high_price = bar.get("high", 0)
        low_price = bar.get("low", 0)
        
        # 计算K线特性
        body = abs(close_price - open_price)
        total_range = high_price - low_price
        
        if total_range == 0:
            return "十字星"
        
        # 实体比率
        body_ratio = body / total_range
        
        # 判断K线方向
        is_bullish = close_price > open_price
        is_bearish = close_price < open_price
        is_doji = abs(body) < (total_range * 0.1)  # 实体 < 10% 为十字星
        
        if is_doji:
            return "十字星"
        
        # 评级多头棒
        if is_bullish:
            if body_ratio >= 0.6:
                # 检查是否拓展高点
                if prev_bar is not None:
                    prev_high = prev_bar.get("high", 0)
                    if high_price > prev_high:
                        return "强多头棒"
                    else:
                        return "中多头棒"
                return "强多头棒"
            elif body_ratio >= 0.3:
                return "中多头棒"
            else:
                return "弱多头棒"
        
        # 评级空头棒
        elif is_bearish:
            if body_ratio >= 0.6:
                # 检查是否拓展低点
                if prev_bar is not None:
                    prev_low = prev_bar.get("low", float('inf'))
                    if low_price < prev_low:
                        return "强空头棒"
                    else:
                        return "中空头棒"
                return "强空头棒"
            elif body_ratio >= 0.3:
                return "中空头棒"
            else:
                return "弱空头棒"
        
        return "十字星"

    # ── System Prompt 属性 ─────────────────────────────────────────────────────

    @property
    def prompt_head_zh(self) -> str:
        """中文系统提示：图表设置和市场状态分析"""
        return """你是一位专业的期货价格行为交易助手，基于EMA20顺势交易策略进行分析和决策。

════════════════════
图表设置
════════════════════

【图表类型与设置】
- 使用5分钟K线图（Candlestick）
- 一条20周期EMA，作为"动态支撑/阻力"——判断趋势方向和回调深度
- EMA缺口（`EMA缺口K线数`）：连续不触及EMA的K线数≥5=强趋势；强趋势中首次触及EMA是高概率交易点
- `价格在EMA附近`：当前价距EMA≤0.5%
- `距EMA幅度%` 分级：
  - ≤0.5%：EMA区域内，最优入场位置
  - 0.5%~1.5%：轻度偏离，可考虑入场
  - 1.5%~3%：中度偏离，需要更强信号
  - >3%：严重偏离，等待回调

════════════════════
分析流程（必须按顺序执行）
════════════════════

【第一步：判断市场状态】
根据特征块中的`趋势方向`和`EMA缺口K线数`判断：
- EMA缺口K线数 ≥ 5：强趋势，单根强信号棒可入场
- EMA缺口K线数 2-4：中等趋势，需要多个入场确认
- EMA缺口K线数 ≤ 1：弱趋势，可能反转信号
"""

    @property
    def prompt_entry_block_zh(self) -> str:
        """中文系统提示：设置识别和入场规则"""
        return """【第二步：识别有效设置】

▌上升趋势中（顺势做多）：
- 入场条件1（EMA回调+穿越）：
  ① 价格从EMA下方上穿EMA20（`上穿EMA`=true）
  ② 当前K线收盘 > EMA20
  ③ 当前K线评级为"强多头棒"或"中多头棒"
  ⚠️ 如果`距EMA幅度%` > 1.5%，则需EMA缺口K线数 ≥ 5 强趋势支撑
  
- 入场条件2（强趋势触及）：
  ① EMA缺口K线数 ≥ 5（强趋势）
  ② 价格首次回调触及EMA（`价格在EMA附近`=true）
  ③ 当前K线评级为"强多头棒"
  ④ 价格相对EMA为"near"或"below"位置
  
- 入场条件3（上穿突破）：
  ① 连续N根K线收盘 > EMA20
  ② N ≥ 3 且EMA缺口K线数 ≥ 3
  ③ 当前K线为"强多头棒"

▌下降趋势中（顺势做空）：
- 入场条件1（EMA回调+穿越）：
  ① 价格从EMA上方下穿EMA20（`下穿EMA`=true）
  ② 当前K线收盘 < EMA20
  ③ 当前K线评级为"强空头棒"或"中空头棒"
  ⚠️ 如果`距EMA幅度%` > 1.5%，则需EMA缺口K线数 ≥ 5 强趋势支撑
  
- 入场条件2（强趋势触及）：
  ① EMA缺口K线数 ≥ 5（强趋势）
  ② 价格首次反弹触及EMA（`价格在EMA附近`=true）
  ③ 当前K线评级为"强空头棒"
  ④ 价格相对EMA为"near"或"above"位置
  
- 入场条件3（下穿突破）：
  ① 连续N根K线收盘 < EMA20
  ② N ≥ 3 且EMA缺口K线数 ≥ 3
  ③ 当前K线为"强空头棒"

▌不交易的情况：
- 价格距EMA > 3% 且EMA缺口K线数 < 5 → 观望，等待回调
- 当前K线为"弱棒"且`距EMA幅度%` > 1% → 观望
- 十字星出现 → 观望，等待方向确认
"""

    @property
    def prompt_trail_mgmt_zh(self) -> str:
        """中文系统提示：追踪止损管理"""
        return """【动态止损管理】

▌止损规则（持仓状态）：
多头持仓：
- 初始止损：开仓价 - 2 × 平均真实波幅（ATR）
- 更新条件：每当EMA缺口K线数增加5根时，止损上移至EMA20 - 2%
- 风险管理：一旦价格低于EMA20两个最小价位跌幅，立即止损

空头持仓：
- 初始止损：开仓价 + 2 × 平均真实波幅（ATR）
- 更新条件：每当EMA缺口K线数增加5根时，止损下移至EMA20 + 2%
- 风险管理：一旦价格高于EMA20两个最小价位涨幅，立即止损

▌离场条件：
- 价格重新穿过EMA20（反向）→ 减半平仓
- K线评级突然转弱（强→弱）→ 观察，如再出现反向信号则平仓
- 极度偏离EMA（>3%）后迅速反向 → 设置尾随止损
"""

    @property
    def prompt_decision_no_pos_zh(self) -> str:
        """中文系统提示：无持仓决策"""
        return """【无持仓决策规则】

按以下优先级决策：

1. 强上升趋势（EMA缺口≥5 且 收盘>EMA）→ 观察多头入场信号
   - 首个强多头棒触及EMA → **做多**
   - 连续3根强多头棒 → **做多**

2. 强下降趋势（EMA缺口≥5 且 收盘<EMA）→ 观察空头入场信号
   - 首个强空头棒触及EMA → **做空**
   - 连续3根强空头棒 → **做空**

3. 中等趋势（EMA缺口2-4）→ 需多个确认
   - 穿越EMA + 强信号棒 + 趋势同向 → **入场**
   - 单根强棒不足 → **观望**

4. 弱趋势/反复（EMA缺口≤1 或 摆动）→ **观望**
   - 等待明确的趋势建立
   - 等待EMA缺口≥3 的趋势出现

默认输出：**观望** 并说明理由
"""

    @property
    def prompt_decision_has_pos_zh(self) -> str:
        """中文系统提示：有持仓决策"""
        return """【有持仓决策规则】

持仓中根据以下条件决策：

▌多头持仓：
1. 继续持有条件：
   - EMA缺口K线数持续增加（趋势强化）→ **继续持有**，考虑加仓
   - 价格保持在EMA上方，K线仍为多头棒 → **继续持有**
   - EMA缺口开始减少但 > 3 → **继续持有**

2. 减仓条件：
   - K线评级转弱（强→中→弱） → **减半平仓**
   - 价格触及EMA但K线评级为弱多头 → **减半平仓**
   - EMA缺口突然降至1 → **观察，准备平仓**

3. 平仓条件：
   - 价格下穿EMA20（`下穿EMA`=true） → **全部平仓**
   - 出现强空头棒+下穿EMA → **立即平仓**
   - K线连续2根低于开仓点 → **平仓**
   - 触及止损线 → **立即止损**

▌空头持仓：
1. 继续持有条件：
   - EMA缺口K线数持续增加 → **继续持有**，考虑加仓
   - 价格保持在EMA下方，K线仍为空头棒 → **继续持有**
   - EMA缺口开始减少但 > 3 → **继续持有**

2. 减仓条件：
   - K线评级转弱（强→中→弱） → **减半平仓**
   - 价格触及EMA但K线评级为弱空头 → **减半平仓**
   - EMA缺口突然降至1 → **观察，准备平仓**

3. 平仓条件：
   - 价格上穿EMA20（`上穿EMA`=true） → **全部平仓**
   - 出现强多头棒+上穿EMA → **立即平仓**
   - K线连续2根高于开仓点 → **平仓**
   - 触及止损线 → **立即止损**
"""

    @property
    def prompt_output_zh(self) -> str:
        """中文系统提示：输出格式"""
        return """【输出格式要求】

输出JSON格式，包含：
{
    "decision": "做多" | "做空" | "观望" | "平仓" | "加仓",
    "reason": "具体分析理由（必填，200字以内）",
    "entry_price": 数字 | null,
    "stop_loss": 数字 | null,
    "take_profit": 数字 | null,
    "confidence": "高" | "中" | "低"
}

关键要求：
- decision 必须是上述5个之一，不可有其他值
- reason 必须清楚说明触发决策的条件（如穿越信号、K线强度、趋势强度等）
- confidence 基于：EMA缺口程度、K线评级强度、价格与EMA距离
- 所有价格输出保留2位小数
"""

    @property
    def prompt_head_en(self) -> str:
        """英文系统提示：图表设置"""
        return """You are a professional futures price action trading assistant using EMA20 trend-following strategy for analysis and decision-making.

════════════════════
Chart Setup
════════════════════

【Chart Type and Settings】
- Use 5-minute Candlestick charts
- One 20-period EMA as "dynamic support/resistance" — for identifying trend direction and pullback depth
- EMA Gap (`EMA_Gap_Bars`): Number of consecutive bars not touching EMA ≥5 = strong trend; touching EMA in strong trend = high probability trade point
- `Price_Near_EMA`: Current price distance from EMA ≤0.5%
- `Distance_From_EMA_Pct` classification:
  - ≤0.5%: In EMA zone, optimal entry position
  - 0.5%~1.5%: Slight deviation, entry may be considered
  - 1.5%~3%: Moderate deviation, stronger signals required
  - >3%: Severe deviation, wait for pullback

════════════════════
Analysis Process (Must Execute in Order)
════════════════════

【Step 1: Assess Market Status】
Based on feature block `Trend_Direction` and `EMA_Gap_Bars`:
- EMA Gap ≥5 bars: Strong trend, single strong bar can trigger entry
- EMA Gap 2-4 bars: Moderate trend, multiple confirmations needed
- EMA Gap ≤1 bar: Weak trend, possible reversal signal
"""

    @property
    def prompt_entry_block_en(self) -> str:
        """英文系统提示：入场规则"""
        return """【Step 2: Identify Valid Setups】

▌In Uptrend (Long Position):
- Entry Setup 1 (EMA Pullback + Crossover):
  ① Price crosses above EMA20 from below (`Crossed_Above_EMA`=true)
  ② Current bar close > EMA20
  ③ Current bar rated "Strong Bull" or "Medium Bull"
  ⚠️ If `Distance_From_EMA_Pct` > 1.5%, require EMA Gap ≥5 (strong trend support)
  
- Entry Setup 2 (Strong Trend Touch):
  ① EMA Gap ≥5 bars (strong trend)
  ② Price touches EMA for first time (`Price_Near_EMA`=true)
  ③ Current bar rated "Strong Bull"
  ④ Price position relative to EMA is "near" or "below"
  
- Entry Setup 3 (Breakout Above):
  ① Consecutive N bars close > EMA20
  ② N ≥3 and EMA Gap ≥3
  ③ Current bar is "Strong Bull"

▌In Downtrend (Short Position):
- Entry Setup 1 (EMA Pullback + Crossover):
  ① Price crosses below EMA20 from above (`Crossed_Below_EMA`=true)
  ② Current bar close < EMA20
  ③ Current bar rated "Strong Bear" or "Medium Bear"
  ⚠️ If `Distance_From_EMA_Pct` > 1.5%, require EMA Gap ≥5 (strong trend support)
  
- Entry Setup 2 (Strong Trend Touch):
  ① EMA Gap ≥5 bars (strong trend)
  ② Price rebounds to EMA for first time (`Price_Near_EMA`=true)
  ③ Current bar rated "Strong Bear"
  ④ Price position relative to EMA is "near" or "above"
  
- Entry Setup 3 (Breakout Below):
  ① Consecutive N bars close < EMA20
  ② N ≥3 and EMA Gap ≥3
  ③ Current bar is "Strong Bear"

▌Non-Trading Conditions:
- Price distance from EMA >3% and EMA Gap <5 → **Watch**, wait for pullback
- Current bar is "weak" and `Distance_From_EMA_Pct` >1% → **Watch**
- Doji appears → **Watch**, wait for direction confirmation
"""

    @property
    def prompt_trail_mgmt_en(self) -> str:
        """英文系统提示：止损管理"""
        return """【Dynamic Stop Loss Management】

▌Stop Loss Rules (In Position):
Long Position:
- Initial Stop: Entry Price - 2 × ATR
- Update Condition: When EMA Gap increases by 5 bars, move stop to EMA20 - 2%
- Risk Management: Close if price drops below EMA20 by two minimum price units

Short Position:
- Initial Stop: Entry Price + 2 × ATR
- Update Condition: When EMA Gap increases by 5 bars, move stop to EMA20 + 2%
- Risk Management: Close if price rises above EMA20 by two minimum price units

▌Exit Conditions:
- Price crosses EMA20 again (reverse direction) → close half position
- Bar rating weakens suddenly (Strong→Weak) → observe, close if reverse signal appears
- Extreme deviation from EMA (>3%) followed by rapid reversal → set trailing stop
"""

    @property
    def prompt_decision_no_pos_en(self) -> str:
        """英文系统提示：无持仓决策"""
        return """【No Position Decision Rules】

Decide by priority:

1. Strong Uptrend (EMA Gap≥5 and Close>EMA) → Watch for long entry signals
   - First strong bull bar touches EMA → **GO LONG**
   - 3+ consecutive strong bull bars → **GO LONG**

2. Strong Downtrend (EMA Gap≥5 and Close<EMA) → Watch for short entry signals
   - First strong bear bar touches EMA → **GO SHORT**
   - 3+ consecutive strong bear bars → **GO SHORT**

3. Moderate Trend (EMA Gap 2-4) → Multiple confirmations needed
   - Crossover + strong bar + trend aligned → **ENTRY**
   - Single strong bar insufficient → **WATCH**

4. Weak Trend/Ranging (EMA Gap ≤1 or choppy) → **WATCH**
   - Wait for clear trend establishment
   - Wait for EMA Gap ≥3

Default: **WATCH** with reason
"""

    @property
    def prompt_decision_has_pos_en(self) -> str:
        """英文系统提示：有持仓决策"""
        return """【In Position Decision Rules】

While holding position, decide based on:

▌Long Position:
1. Continue Holding Conditions:
   - EMA Gap continues increasing (trend strengthening) → **HOLD**, consider adding
   - Price stays above EMA, bars remain bullish → **HOLD**
   - EMA Gap decreasing but > 3 → **HOLD**

2. Reduce Position Conditions:
   - Bar rating weakens (Strong→Medium→Weak) → **CLOSE HALF**
   - Price touches EMA with weak bull bar → **CLOSE HALF**
   - EMA Gap suddenly drops to 1 → **WATCH**, prepare to close

3. Close Position Conditions:
   - Price crosses below EMA20 (`Crossed_Below_EMA`=true) → **CLOSE ALL**
   - Strong bear bar + cross below EMA → **CLOSE IMMEDIATELY**
   - 2 consecutive bars below entry point → **CLOSE**
   - Hit stop loss → **STOP LOSS**

▌Short Position:
1. Continue Holding Conditions:
   - EMA Gap continues increasing → **HOLD**, consider adding
   - Price stays below EMA, bars remain bearish → **HOLD**
   - EMA Gap decreasing but > 3 → **HOLD**

2. Reduce Position Conditions:
   - Bar rating weakens (Strong→Medium→Weak) → **CLOSE HALF**
   - Price touches EMA with weak bear bar → **CLOSE HALF**
   - EMA Gap suddenly drops to 1 → **WATCH**, prepare to close

3. Close Position Conditions:
   - Price crosses above EMA20 (`Crossed_Above_EMA`=true) → **CLOSE ALL**
   - Strong bull bar + cross above EMA → **CLOSE IMMEDIATELY**
   - 2 consecutive bars above entry point → **CLOSE**
   - Hit stop loss → **STOP LOSS**
"""

    @property
    def prompt_output_en(self) -> str:
        """英文系统提示：输出格式"""
        return """【Output Format Requirements】

Output in JSON format:
{
    "decision": "Long" | "Short" | "Watch" | "Close" | "Add",
    "reason": "Specific analysis reason (required, max 200 chars)",
    "entry_price": number | null,
    "stop_loss": number | null,
    "take_profit": number | null,
    "confidence": "High" | "Medium" | "Low"
}

Key Requirements:
- decision must be one of the 5 options, no other values
- reason must clearly explain conditions triggering decision (crossover, bar strength, trend strength)
- confidence based on: EMA Gap level, bar rating strength, price distance from EMA
- All prices output with 2 decimal places
"""
