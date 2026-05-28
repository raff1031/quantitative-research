#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using NinjaTrader.Cbi;
using NinjaTrader.Gui.Tools;
using NinjaTrader.NinjaScript;
using NinjaTrader.NinjaScript.Strategies;
#endregion

// ============================================================================
// ORB_MultiOpenRangeBreakout_Safe (M1, no tick, SAFE)
// - Più finestre OR via WindowsSpec: "offsetxlength, ...", es. "10x15"
// - FIX: l’OR non include la barra con Time == EndTime (off-by-one corretto)
// - FIX: ingresso SEMPRE alla barra successiva alla conferma (anche in storico)
// - Direzione: Both / LongOnly / ShortOnly
// - Flat manuale sull’ultima barra di sessione
// ============================================================================
namespace NinjaTrader.NinjaScript.Strategies
{
    public class ORB_MultiOpenRangeBreakout_Safe : Strategy
    {
        public enum TradeDirection { Both, LongOnly, ShortOnly }

        // -------------------- Input --------------------
        [NinjaScriptProperty]
        [Display(Name = "WindowsSpec", Order = 0, GroupName = "Windows",
            Description = "offset x length in minuti dal session open. Es: 0x60,10x15,60x30")]
        public string WindowsSpec { get; set; }

        [NinjaScriptProperty, Range(0, 20)]
        [Display(Name = "EntryBufferTicks", Order = 2, GroupName = "Parameters")]
        public int EntryBufferTicks { get; set; }

        [NinjaScriptProperty, Range(0, 20)]
        [Display(Name = "StopBufferTicks", Order = 3, GroupName = "Parameters")]
        public int StopBufferTicks { get; set; }

        [NinjaScriptProperty, Range(0.0, 5.0)]
        [Display(Name = "ProfitTarget_OR_Multiple", Order = 4, GroupName = "Parameters",
            Description = "0 = nessun PT; altrimenti multiplo della larghezza dell’OR")]
        public double ProfitTargetOrMultiple { get; set; }

        [NinjaScriptProperty, Range(0, 10000)]
        [Display(Name = "MinORRangeTicks", Order = 5, GroupName = "Filters")]
        public int MinORRangeTicks { get; set; }

        [NinjaScriptProperty, Range(0, 10000)]
        [Display(Name = "MaxORRangeTicks", Order = 6, GroupName = "Filters")]
        public int MaxORRangeTicks { get; set; }

        [NinjaScriptProperty, Range(1, 100)]
        [Display(Name = "Contracts", Order = 7, GroupName = "Risk")]
        public int Contracts { get; set; }

        [NinjaScriptProperty, Range(1, 10)]
        [Display(Name = "MaxTradesPerSession", Order = 8, GroupName = "Behavior",
            Description = "Numero massimo di trade al giorno (tutte le finestre)")]
        public int MaxTradesPerSession { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "UseCloseConfirmation", Order = 10, GroupName = "Behavior",
            Description = "TRUE = richiede chiusura oltre il trigger; ingresso alla barra successiva")]
        public bool UseCloseConfirmation { get; set; }

        [NinjaScriptProperty, Range(1, 5)]
        [Display(Name = "ConfirmBars", Order = 11, GroupName = "Behavior")]
        public int ConfirmBars { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "SkipIfBothSidesTouched", Order = 12, GroupName = "Filters",
            Description = "Se una barra tocca entrambi i trigger, skip (solo quando Direction=Both)")]
        public bool SkipIfBothSidesTouched { get; set; }

        [NinjaScriptProperty]
        [Display(Name = "Direction", Order = 13, GroupName = "Behavior")]
        public TradeDirection Direction { get; set; }

        // -------------------- Internals --------------------
        private DateTime sessionStart;
        private int tradesThisSession;
        private List<OrWindow> windows;

        private class OrWindow
        {
            public int Index;
            public int OffsetMinutes;
            public int LengthMinutes;

            public DateTime StartTime;
            public DateTime EndTime;

            public bool InOR;
            public bool OrComplete;
            public bool Signaled;
            public bool Traded;

            public double OrHigh;
            public double OrLow;

            // SAFE: ingresso alla barra successiva
            public bool PendingLongNextBar;
            public bool PendingShortNextBar;
            public int  SignalBarIndex;   // indice della barra su cui nasce la conferma

            public string LongTag;
            public string ShortTag;
        }

        protected override void OnStateChange()
        {
            if (State == State.SetDefaults)
            {
                Name = "ORB_MultiOpenRangeBreakout_Safe";
                Calculate = Calculate.OnBarClose;
                EntriesPerDirection = 1;
                EntryHandling = EntryHandling.AllEntries;
                IsUnmanaged = false;
                IsInstantiatedOnEachOptimizationIteration = false;

                WindowsSpec = "0x60";
                EntryBufferTicks = 1;
                StopBufferTicks = 1;
                ProfitTargetOrMultiple = 1.0;
                MinORRangeTicks = 0;
                MaxORRangeTicks = 0;
                Contracts = 1;
                MaxTradesPerSession = 1;

                UseCloseConfirmation = true;
                ConfirmBars = 1;
                SkipIfBothSidesTouched = true;
                Direction = TradeDirection.Both;

                windows = new List<OrWindow>();
            }
            else if (State == State.DataLoaded)
            {
                windows = ParseWindows(WindowsSpec);
            }
        }

        protected override void OnBarUpdate()
        {
            if (CurrentBar < 5) return;

            if (Bars.IsFirstBarOfSession)
                StartNewSession();

            // Chiudi tutto a fine sessione
            if (Bars.IsLastBarOfSession && Position.MarketPosition != MarketPosition.Flat)
            {
                if (Position.MarketPosition == MarketPosition.Long)  ExitLong();
                else if (Position.MarketPosition == MarketPosition.Short) ExitShort();
            }

            if (tradesThisSession >= MaxTradesPerSession)
                return;

            bool allowLong  = Direction != TradeDirection.ShortOnly;
            bool allowShort = Direction != TradeDirection.LongOnly;

            for (int i = 0; i < windows.Count; i++)
            {
                var w = windows[i];

                // Avvio finestra OR quando entro nell’intervallo
                if (!w.InOR && !w.OrComplete && Time[0] >= w.StartTime && Time[0] < w.EndTime)
                {
                    w.InOR = true;
                    w.OrHigh = double.MinValue;
                    w.OrLow  = double.MaxValue;
                }

                // COSTRUZIONE OR (EndTime esclusivo, FIX off-by-one)
                if (w.InOR)
                {
                    // se siamo già oltre o uguali a EndTime, chiudi la finestra senza aggiornare
                    if (Time[0] >= w.EndTime)
                    {
                        w.InOR = false;
                        w.OrComplete = true;
                    }
                    else
                    {
                        // includi solo barre con Time < EndTime
                        w.OrHigh = Math.Max(w.OrHigh, High[0]);
                        w.OrLow  = Math.Min(w.OrLow,  Low[0]);
                    }
                    continue;
                }

                // Post-OR: genera segnale
                if (w.OrComplete && !w.Traded && !w.Signaled && Position.MarketPosition == MarketPosition.Flat)
                {
                    double rangeTicks = (w.OrHigh - w.OrLow) / TickSize;
                    if ((MinORRangeTicks > 0 && rangeTicks < MinORRangeTicks) ||
                        (MaxORRangeTicks > 0 && rangeTicks > MaxORRangeTicks))
                    { w.Traded = true; continue; }

                    double longTrigger  = w.OrHigh + EntryBufferTicks * TickSize;
                    double shortTrigger = w.OrLow  - EntryBufferTicks * TickSize;

                    bool bothTouched = (Direction == TradeDirection.Both) &&
                                       High[0] >= longTrigger && Low[0] <= shortTrigger;
                    if (SkipIfBothSidesTouched && bothTouched)
                    { w.Traded = true; continue; }

                    double longStopPrice  = w.OrLow  - StopBufferTicks * TickSize;
                    double shortStopPrice = w.OrHigh + StopBufferTicks * TickSize;

                    if (ProfitTargetOrMultiple > 0.0)
                    {
                        double orRange = w.OrHigh - w.OrLow;
                        if (allowLong)
                            SetProfitTarget(w.LongTag,  CalculationMode.Price, w.OrHigh + ProfitTargetOrMultiple * orRange);
                        if (allowShort)
                            SetProfitTarget(w.ShortTag, CalculationMode.Price, w.OrLow  - ProfitTargetOrMultiple * orRange);
                    }
                    else
                    {
                        if (allowLong)  SetProfitTarget(w.LongTag,  CalculationMode.Ticks, int.MaxValue);
                        if (allowShort) SetProfitTarget(w.ShortTag, CalculationMode.Ticks, int.MaxValue);
                    }

                    if (allowLong)  SetStopLoss(w.LongTag,  CalculationMode.Price, longStopPrice,  false);
                    if (allowShort) SetStopLoss(w.ShortTag, CalculationMode.Price, shortStopPrice, false);

                    if (UseCloseConfirmation)
                    {
                        bool longOK = false, shortOK = false;
                        int checks = Math.Min(ConfirmBars, CurrentBar + 1);

                        if (allowLong)
                        {
                            longOK = true;
                            for (int b = 0; b < checks; b++)
                                if (Close[b] <= longTrigger) { longOK = false; break; }
                        }
                        if (allowShort)
                        {
                            shortOK = true;
                            for (int b = 0; b < checks; b++)
                                if (Close[b] >= shortTrigger) { shortOK = false; break; }
                        }

                        if (longOK || shortOK)
                        {
                            w.PendingLongNextBar  = longOK;
                            w.PendingShortNextBar = (!longOK && shortOK);
                            w.Signaled = true;
                            w.SignalBarIndex = CurrentBar; // **ingresso dalla barra successiva**
                        }
                    }
                    else
                    {
                        // TOUCH MODE (OCO) — sconsigliato senza tick
                        // if (allowLong)  EnterLongStopMarket(0, true, Contracts, longTrigger,  w.LongTag);
                        // if (allowShort) EnterShortStopMarket(0, true, Contracts, shortTrigger, w.ShortTag);
                        // w.Signaled = true; w.SignalBarIndex = CurrentBar;
                    }
                }

                // SAFE: ingresso **solo** quando siamo su una barra successiva al segnale
                if (UseCloseConfirmation && w.Signaled && !w.Traded && Position.MarketPosition == MarketPosition.Flat)
                {
                    if (CurrentBar > w.SignalBarIndex)
                    {
                        if (w.PendingLongNextBar)
                        {
                            EnterLong(Contracts, w.LongTag);
                            w.PendingLongNextBar  = false; w.PendingShortNextBar = false;
                            w.Traded = true; tradesThisSession++;
                        }
                        else if (w.PendingShortNextBar)
                        {
                            EnterShort(Contracts, w.ShortTag);
                            w.PendingShortNextBar = false; w.PendingLongNextBar  = false;
                            w.Traded = true; tradesThisSession++;
                        }
                    }
                }

                if (tradesThisSession >= MaxTradesPerSession) break;
            }
        }

        private void StartNewSession()
        {
            sessionStart = Time[0];
            tradesThisSession = 0;

            foreach (var w in windows)
            {
                w.StartTime = sessionStart.AddMinutes(w.OffsetMinutes);
                w.EndTime   = w.StartTime.AddMinutes(w.LengthMinutes);

                w.InOR = false;
                w.OrComplete = false;
                w.Signaled = false;
                w.Traded = false;

                w.OrHigh = double.MinValue;
                w.OrLow  = double.MaxValue;

                w.PendingLongNextBar  = false;
                w.PendingShortNextBar = false;
                w.SignalBarIndex = -1;
            }

            // Se la prima barra cade dentro una finestra, inizia subito l'OR
            foreach (var w in windows)
            {
                if (Time[0] >= w.StartTime && Time[0] < w.EndTime)
                {
                    w.InOR = true;
                    w.OrHigh = double.MinValue;
                    w.OrLow  = double.MaxValue;
                }
            }
        }

        private List<OrWindow> ParseWindows(string spec)
        {
            var list = new List<OrWindow>();
            if (string.IsNullOrWhiteSpace(spec)) return list;

            string[] parts = spec.Split(new[] { ',' }, StringSplitOptions.RemoveEmptyEntries);
            int idx = 0;
            foreach (var raw in parts)
            {
                var s = raw.Trim().ToLower();
                char sep = s.Contains("x") ? 'x' : (s.Contains(":") ? ':' : '\0');
                if (sep == '\0') continue;

                string[] duo = s.Split(sep);
                if (duo.Length != 2) continue;

                if (!int.TryParse(duo[0], out int offset)) continue;
                if (!int.TryParse(duo[1], out int length)) continue;
                if (length < 1) continue;

                list.Add(new OrWindow {
                    Index = idx,
                    OffsetMinutes = offset,
                    LengthMinutes = length,
                    LongTag = $"LBreak_{idx}",
                    ShortTag = $"SBreak_{idx}"
                });
                idx++;
            }
            return list;
        }

        // (Override per TOUCH MODE, se mai ti servisse; lasciali commentati)
        /*
        protected override void OnOrderUpdate(NinjaTrader.Cbi.Order order,
                                              double limitPrice, double stopPrice,
                                              int quantity, int filled, double averageFillPrice,
                                              NinjaTrader.Cbi.OrderState orderState, DateTime time,
                                              NinjaTrader.Cbi.ErrorCode error, string nativeError) { }

        protected override void OnExecutionUpdate(NinjaTrader.Cbi.Execution execution,
                                                  NinjaTrader.Cbi.Order order) { }
        */
    }
}
