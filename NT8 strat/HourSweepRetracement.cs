
#region Using declarations
using System;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Data;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.Strategies;
#endregion

// Strategy: HourSweepRetracement
// Versione: con filtro trend Bollinger e break-even
// - Fuso del grafico per aggregare open/H/L orari; conversione ET solo per filtri temporali
// - Sweep vs range dell'ora precedente (open inside), ingresso verso l'open
// - Limiti: 1 trade/segmento (0-20/20-40/40-60) e MaxTradesPerSession
// - Exit: TP/SL ATR opzionali, oppure target all'open + SL a ticks
// - NUOVO: Trend filter Bollinger (Close>Mid => SHORT permessi; Close<Mid => LONG permessi)
// - NUOVO: Opzione BreakEven: porta lo stop a prezzo di ingresso dopo X ticks di favore

namespace NinjaTrader.NinjaScript.Strategies
{
    public class HourSweepRetracement : Strategy
    {
        // ---- Timezones ----
        private TimeZoneInfo chartTz;
        private TimeZoneInfo easternTz;

        // ---- Stato orario aggregato ----
        private DateTime currentHourKey = DateTime.MinValue;
        private double currentHourOpen  = 0.0;
        private double currentHourHigh  = 0.0;
        private double currentHourLow   = 0.0;
        private double prevHourHigh     = 0.0;
        private double prevHourLow      = 0.0;

        private bool sweepOccurredThisHour = false;

        // ---- Stato di sessione ET ----
        private DateTime sessionDateET = DateTime.MinValue;
        private int tradesThisSession = 0;
        private int tradesSeg0Today = 0;
        private int tradesSeg1Today = 0;
        private int tradesSeg2Today = 0;

        // ---- Indicatori ----
        private Bollinger bb;

        // ---- Break-even ----
        private bool movedToBE = false;

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name                                    = "HourSweepRetracement";
                Description                             = "Sweep+Return-to-Open orario; BB trend filter e break-even opzionali.";
                Calculate                               = Calculate.OnEachTick;
                EntriesPerDirection                     = 1;
                EntryHandling                           = EntryHandling.AllEntries;
                IsExitOnSessionCloseStrategy            = true;
                ExitOnSessionCloseSeconds               = 30;
                Slippage                                = 0;
                StartBehavior                           = StartBehavior.WaitUntilFlat;
                TimeInForce                             = TimeInForce.Gtc;
                TraceOrders                             = false;
                RealtimeErrorHandling                   = RealtimeErrorHandling.StopCancelClose;
                StopTargetHandling                      = StopTargetHandling.PerEntryExecution;
                BarsRequiredToTrade                     = 5;

                // Timing & TZ
                DebugMode               = false;
                StartHourET             = 9;
                EndHourET               = 16;
                TradeFirstSegment       = true;
                TradeSecondSegment      = true;
                TradeThirdSegment       = true;
                ChartTimeZoneStr        = "Central European Standard Time";

                // Risk & Exit
                StopLossTicks           = 40;
                UseATRExit              = true;
                ATRPeriod               = 14;
                ATR_TP_Mult             = 1.0;
                ATR_SL_Mult             = 1.0;

                // Limits
                OnePerSegment           = true;
                MaxTradesPerSession     = 3;

                // Bollinger filter
                UseBBTrendFilter        = true;
                BBPeriod                = 20;
                BBStdDev                = 2.0;

                // Break-even
                UseBreakEven            = true;
                BreakEvenTicks          = 5;
            }
            else if (State == State.Configure)
            {
                try
                {
                    chartTz   = TimeZoneInfo.FindSystemTimeZoneById(ChartTimeZoneStr);
                    easternTz = TimeZoneInfo.FindSystemTimeZoneById("Eastern Standard Time");
                    if (DebugMode)
                        Print($"[Init] Chart TZ: {chartTz.DisplayName} | ET: {easternTz.DisplayName}");
                }
                catch (Exception e)
                {
                    Print("Timezone error: " + e.Message);
                }
            }
            else if (State == State.DataLoaded)
            {
                if (UseBBTrendFilter)
                {
                    bb = Bollinger(BBStdDev, BBPeriod); // NT8 signature: Bollinger(double numStdDev, int period)
                    // AddChartIndicator(bb); // opzionale per visualizzare
                }
            }
        }

        protected override void OnBarUpdate()
        {
            int minBars = Math.Max(BarsRequiredToTrade, UseATRExit ? (ATRPeriod + 1) : BarsRequiredToTrade);
            if (CurrentBar < minBars)
                return;

            // Tempo locale grafico
            DateTime tLocal = Time[0];
            DateTime hourKey = new DateTime(tLocal.Year, tLocal.Month, tLocal.Day, tLocal.Hour, 0, 0);

            // Cambio ora: roll e reset
            if (currentHourKey == DateTime.MinValue)
            {
                currentHourKey = hourKey;
                currentHourOpen = Open[0];
                currentHourHigh = High[0];
                currentHourLow  = Low[0];
                return;
            }
            else if (hourKey != currentHourKey)
            {
                if (Position.MarketPosition != MarketPosition.Flat)
                {
                    if (DebugMode) Print($"{tLocal:yyyy-MM-dd HH:mm} Local | Cambio ora -> Exit flat");
                    if (Position.MarketPosition == MarketPosition.Long)
                        ExitLong("Fine Ora", "EnterLong");
                    else if (Position.MarketPosition == MarketPosition.Short)
                        ExitShort("Fine Ora", "EnterShort");
                }

                prevHourHigh = currentHourHigh;
                prevHourLow  = currentHourLow;

                currentHourKey  = hourKey;
                currentHourOpen = Open[0];
                currentHourHigh = High[0];
                currentHourLow  = Low[0];
                sweepOccurredThisHour = false;
                movedToBE = false;

                return;
            }
            else
            {
                if (High[0] > currentHourHigh) currentHourHigh = High[0];
                if (Low[0]  < currentHourLow)  currentHourLow  = Low[0];
            }

            // Reset BE quando flat
            if (Position.MarketPosition == MarketPosition.Flat && movedToBE)
                movedToBE = false;

            // Serve una previous hour valida
            if (prevHourHigh == 0.0 && prevHourLow == 0.0)
                return;

            // Filtro ET finestra / segmenti
            DateTime tET = TimeZoneInfo.ConvertTime(tLocal, chartTz, easternTz);
            int minuteOfHour = tET.Minute;
            bool inEtWindow = tET.Hour >= StartHourET && tET.Hour < EndHourET;
            if (!inEtWindow) return;

            // Reset contatori a nuovo giorno ET
            if (sessionDateET != tET.Date)
            {
                sessionDateET = tET.Date;
                tradesThisSession = 0;
                tradesSeg0Today = tradesSeg1Today = tradesSeg2Today = 0;
                if (DebugMode) Print($"ET {tET:yyyy-MM-dd} | Reset contatori (MaxTradesPerSession={MaxTradesPerSession}, OnePerSegment={OnePerSegment})");
            }

            int seg = (minuteOfHour < 20) ? 0 : (minuteOfHour < 40 ? 1 : 2);
            bool segEnabled =
                (seg == 0 && TradeFirstSegment) ||
                (seg == 1 && TradeSecondSegment) ||
                (seg == 2 && TradeThirdSegment);

            if (!segEnabled)
                return;

            if (Position.MarketPosition != MarketPosition.Flat || sweepOccurredThisHour)
                return;

            if (tradesThisSession >= MaxTradesPerSession)
            {
                if (DebugMode && IsFirstTickOfBar) Print($"ET {tET:HH:mm} | SKIP: MaxTradesPerSession.");
                return;
            }
            if (OnePerSegment)
            {
                if ((seg == 0 && tradesSeg0Today >= 1) ||
                    (seg == 1 && tradesSeg1Today >= 1) ||
                    (seg == 2 && tradesSeg2Today >= 1))
                {
                    if (DebugMode && IsFirstTickOfBar) Print($"ET {tET:HH:mm} | SKIP: già 1 trade nel segmento {seg} oggi.");
                    return;
                }
            }

            // ---- Bollinger trend filter ----
            bool allowShortByBB = true;
            bool allowLongByBB  = true;
            if (UseBBTrendFilter && bb != null && bb.Middle != null)
            {
                double mid = bb.Middle[0];
                // Regola richiesta: se Close > mid => SELL (short) validi; se Close < mid => LONG validi
                allowShortByBB = Close[0] > mid;
                allowLongByBB  = Close[0] < mid;
            }

            // ---- Condizione INSIDE ----
            bool isInsideOpen = (currentHourOpen < prevHourHigh && currentHourOpen > prevHourLow);
            if (DebugMode && IsFirstTickOfBar)
                Print($"ET {tET:HH:mm} | OpenHour={currentHourOpen:F2} PrevH={prevHourHigh:F2} PrevL={prevHourLow:F2} Inside={isInsideOpen} | BB mid={(bb!=null?bb.Middle[0].ToString("F2"):"na")} AllowS={allowShortByBB} AllowL={allowLongByBB}");

            if (!isInsideOpen)
                return;

            // ---- ATR in ticks per TP/SL ----
            double atrTicks = 0.0;
            if (UseATRExit)
            {
                double atr = ATR(ATRPeriod)[0];
                atrTicks = atr / TickSize;
                if (atrTicks <= 0) return;
            }

            // ---- SWEEP e ingresso ----
            bool entered = false;
            if (High[0] > prevHourHigh && allowShortByBB)
            {
                if (DebugMode) Print($"ET {tET:HH:mm} | SWEEP UP -> SHORT");
                if (UseATRExit)
                {
                    SetProfitTarget("EnterShort", CalculationMode.Ticks, Math.Max(1, (int)Math.Round(ATR_TP_Mult * atrTicks)));
                    SetStopLoss   ("EnterShort", CalculationMode.Ticks, Math.Max(1, (int)Math.Round(ATR_SL_Mult * atrTicks)), false);
                }
                else
                {
                    SetProfitTarget("EnterShort", CalculationMode.Price, currentHourOpen);
                    SetStopLoss   ("EnterShort", CalculationMode.Ticks, StopLossTicks, false);
                }
                EnterShort(DefaultQuantity, "EnterShort");
                entered = true;
            }
            else if (Low[0] < prevHourLow && allowLongByBB)
            {
                if (DebugMode) Print($"ET {tET:HH:mm} | SWEEP DOWN -> LONG");
                if (UseATRExit)
                {
                    SetProfitTarget("EnterLong", CalculationMode.Ticks, Math.Max(1, (int)Math.Round(ATR_TP_Mult * atrTicks)));
                    SetStopLoss   ("EnterLong", CalculationMode.Ticks, Math.Max(1, (int)Math.Round(ATR_SL_Mult * atrTicks)), false);
                }
                else
                {
                    SetProfitTarget("EnterLong", CalculationMode.Price, currentHourOpen);
                    SetStopLoss   ("EnterLong", CalculationMode.Ticks, StopLossTicks, false);
                }
                EnterLong(DefaultQuantity, "EnterLong");
                entered = true;
            }

            if (entered)
            {
                sweepOccurredThisHour = true;
                tradesThisSession += 1;
                if      (seg == 0) tradesSeg0Today += 1;
                else if (seg == 1) tradesSeg1Today += 1;
                else if (seg == 2) tradesSeg2Today += 1;
                movedToBE = false;
                if (DebugMode) Print($"ET {tET:HH:mm} | Trade registrato. Totale oggi: {tradesThisSession} (seg0={tradesSeg0Today}, seg1={tradesSeg1Today}, seg2={tradesSeg2Today})");
            }

            // ---- Break-even management ----
            if (UseBreakEven && Position.MarketPosition != MarketPosition.Flat && !movedToBE)
            {
                double beTicks = Math.Max(1, BreakEvenTicks);
                double beDist = beTicks * TickSize;

                if (Position.MarketPosition == MarketPosition.Long)
                {
                    if (Close[0] - Position.AveragePrice >= beDist)
                    {
                        SetStopLoss("EnterLong", CalculationMode.Price, Position.AveragePrice, false);
                        movedToBE = true;
                        if (DebugMode) Print($"ET {tET:HH:mm} | LONG -> stop a BE ({Position.AveragePrice:F2})");
                    }
                }
                else if (Position.MarketPosition == MarketPosition.Short)
                {
                    if (Position.AveragePrice - Close[0] >= beDist)
                    {
                        SetStopLoss("EnterShort", CalculationMode.Price, Position.AveragePrice, false);
                        movedToBE = true;
                        if (DebugMode) Print($"ET {tET:HH:mm} | SHORT -> stop a BE ({Position.AveragePrice:F2})");
                    }
                }
            }
        }

        #region Properties
        [NinjaScriptProperty]
        [Display(Name = "Debug Mode", Order = 0, GroupName = "Parameters")]
        public bool DebugMode { get; set; }

        [NinjaScriptProperty]
        [Range(0, 23)]
        [Display(Name = "Ora Inizio (ET)", Order = 1, GroupName = "Timing")]
        public int StartHourET { get; set; }

        [NinjaScriptProperty]
        [Range(1, 24)]
        [Display(Name = "Ora Fine (ET)", Order = 2, GroupName = "Timing")]
        public int EndHourET { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade 0-20 min", Order = 3, GroupName = "Timing")]
        public bool TradeFirstSegment { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade 20-40 min", Order = 4, GroupName = "Timing")]
        public bool TradeSecondSegment { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Trade 40-60 min", Order = 5, GroupName = "Timing")]
        public bool TradeThirdSegment { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Chart TimeZone ID", Description = "Imposta il fuso del grafico (es. 'Central European Standard Time' per Roma).", Order = 6, GroupName = "Parameters")]
        public string ChartTimeZoneStr { get; set; }

        // --- Risk & Exit ---
        [NinjaScriptProperty]
        [Range(1, int.MaxValue)]
        [Display(Name = "Stop Loss (Ticks) se non ATR", Order = 7, GroupName = "Risk/Exit")]
        public int StopLossTicks { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Usa TP/SL ATR", Order = 8, GroupName = "Risk/Exit")]
        public bool UseATRExit { get; set; }

        [NinjaScriptProperty]
        [Range(2, 200)]
        [Display(Name = "ATR Period", Order = 9, GroupName = "Risk/Exit")]
        public int ATRPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 10.0)]
        [Display(Name = "ATR TP Mult", Order = 10, GroupName = "Risk/Exit")]
        public double ATR_TP_Mult { get; set; }

        [NinjaScriptProperty]
        [Range(0.1, 10.0)]
        [Display(Name = "ATR SL Mult", Order = 11, GroupName = "Risk/Exit")]
        public double ATR_SL_Mult { get; set; }

        // --- Limits ---
        [NinjaScriptProperty]
        [Display(Name = "1 trade per segmento (giorno ET)", Order = 12, GroupName = "Limits")]
        public bool OnePerSegment { get; set; }

        [NinjaScriptProperty]
        [Range(1, 20)]
        [Display(Name = "Max trades per sessione (ET)", Order = 13, GroupName = "Limits")]
        public int MaxTradesPerSession { get; set; }

        // --- Bollinger filter ---
        [NinjaScriptProperty]
        [Display(Name = "Usa filtro Bollinger", Order = 14, GroupName = "Trend Filter")]
        public bool UseBBTrendFilter { get; set; }

        [NinjaScriptProperty]
        [Range(5, 200)]
        [Display(Name = "BB Period", Order = 15, GroupName = "Trend Filter")]
        public int BBPeriod { get; set; }

        [NinjaScriptProperty]
        [Range(0.5, 5.0)]
        [Display(Name = "BB StdDev", Order = 16, GroupName = "Trend Filter")]
        public double BBStdDev { get; set; }

        // --- Break-even ---
        [NinjaScriptProperty]
        [Display(Name = "Usa BreakEven", Order = 17, GroupName = "BreakEven")]
        public bool UseBreakEven { get; set; }

        [NinjaScriptProperty]
        [Range(1, 50)]
        [Display(Name = "BreakEven Ticks", Order = 18, GroupName = "BreakEven")]
        public int BreakEvenTicks { get; set; }
        #endregion
    }
}

#region NinjaScript generated code. Neither change nor remove.
namespace NinjaTrader.NinjaScript.Strategies
{
    public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
    {
        private HourSweepRetracement[] cacheHourSweepRetracement;
        public HourSweepRetracement HourSweepRetracement(bool debugMode, int startHourET, int endHourET, bool tradeFirstSegment, bool tradeSecondSegment, bool tradeThirdSegment, string chartTimeZoneStr, int stopLossTicks, bool useATRExit, int aTRPeriod, double aTR_TP_Mult, double aTR_SL_Mult, bool onePerSegment, int maxTradesPerSession, bool useBBTrendFilter, int bBPeriod, double bBStdDev, bool useBreakEven, int breakEvenTicks)
        {
            return HourSweepRetracement(Input, debugMode, startHourET, endHourET, tradeFirstSegment, tradeSecondSegment, tradeThirdSegment, chartTimeZoneStr, stopLossTicks, useATRExit, aTRPeriod, aTR_TP_Mult, aTR_SL_Mult, onePerSegment, maxTradesPerSession, useBBTrendFilter, bBPeriod, bBStdDev, useBreakEven, breakEvenTicks);
        }

        public HourSweepRetracement HourSweepRetracement(ISeries<double> input, bool debugMode, int startHourET, int endHourET, bool tradeFirstSegment, bool tradeSecondSegment, bool tradeThirdSegment, string chartTimeZoneStr, int stopLossTicks, bool useATRExit, int aTRPeriod, double aTR_TP_Mult, double aTR_SL_Mult, bool onePerSegment, int maxTradesPerSession, bool useBBTrendFilter, int bBPeriod, double bBStdDev, bool useBreakEven, int breakEvenTicks)
        {
            if (cacheHourSweepRetracement != null)
                for (int idx = 0; idx < cacheHourSweepRetracement.Length; idx++)
                    if (cacheHourSweepRetracement[idx] != null && cacheHourSweepRetracement[idx].DebugMode == debugMode
                        && cacheHourSweepRetracement[idx].StartHourET == startHourET
                        && cacheHourSweepRetracement[idx].EndHourET == endHourET
                        && cacheHourSweepRetracement[idx].TradeFirstSegment == tradeFirstSegment
                        && cacheHourSweepRetracement[idx].TradeSecondSegment == tradeSecondSegment
                        && cacheHourSweepRetracement[idx].TradeThirdSegment == tradeThirdSegment
                        && cacheHourSweepRetracement[idx].ChartTimeZoneStr == chartTimeZoneStr
                        && cacheHourSweepRetracement[idx].StopLossTicks == stopLossTicks
                        && cacheHourSweepRetracement[idx].UseATRExit == useATRExit
                        && cacheHourSweepRetracement[idx].ATRPeriod == aTRPeriod
                        && Math.Abs(cacheHourSweepRetracement[idx].ATR_TP_Mult - aTR_TP_Mult) < 1e-9
                        && Math.Abs(cacheHourSweepRetracement[idx].ATR_SL_Mult - aTR_SL_Mult) < 1e-9
                        && cacheHourSweepRetracement[idx].OnePerSegment == onePerSegment
                        && cacheHourSweepRetracement[idx].MaxTradesPerSession == maxTradesPerSession
                        && cacheHourSweepRetracement[idx].UseBBTrendFilter == useBBTrendFilter
                        && cacheHourSweepRetracement[idx].BBPeriod == bBPeriod
                        && Math.Abs(cacheHourSweepRetracement[idx].BBStdDev - bBStdDev) < 1e-9
                        && cacheHourSweepRetracement[idx].UseBreakEven == useBreakEven
                        && cacheHourSweepRetracement[idx].BreakEvenTicks == breakEvenTicks
                        && cacheHourSweepRetracement[idx].EqualsInput(input))
                        return cacheHourSweepRetracement[idx];
            return new HourSweepRetracement(){ DebugMode = debugMode, StartHourET = startHourET, EndHourET = endHourET, TradeFirstSegment = tradeFirstSegment, TradeSecondSegment = tradeSecondSegment, TradeThirdSegment = tradeThirdSegment, ChartTimeZoneStr = chartTimeZoneStr, StopLossTicks = stopLossTicks, UseATRExit = useATRExit, ATRPeriod = aTRPeriod, ATR_TP_Mult = aTR_TP_Mult, ATR_SL_Mult = aTR_SL_Mult, OnePerSegment = onePerSegment, MaxTradesPerSession = maxTradesPerSession, UseBBTrendFilter = useBBTrendFilter, BBPeriod = bBPeriod, BBStdDev = bBStdDev, UseBreakEven = useBreakEven, BreakEvenTicks = breakEvenTicks };
        }
    }
}
#endregion
