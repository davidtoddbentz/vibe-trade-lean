using System;
using QuantConnect.Data;
using QuantConnect.Logging;
using QuantConnect.Orders;
using QuantConnect.Orders.Fills;
using QuantConnect.Securities;

namespace QuantConnect.Lean.Engine.FillModels
{
    /// <summary>
    /// Fill model that reads OHLC from PythonData's DynamicData storage
    /// instead of Security.High/Low (which always equal Close for PythonData).
    ///
    /// PythonData.Reader() stores OHLC via bracket notation (data["High"] = value).
    /// DynamicData.SetProperty() lowercases keys, so storage has "high", "low", etc.
    /// This fill model reads those values via GetProperty() which handles case conversion.
    ///
    /// With correct OHLC, LEAN's native LimitFill/StopMarketFill/StopLimitFill
    /// evaluate fill conditions and fill at correct prices.
    /// </summary>
    public class CustomDataFillModel : FillModel
    {
        private bool _loggedOnce = false;

        /// <summary>
        /// Override GetPrices to read OHLC from PythonData's DynamicData storage.
        /// </summary>
        protected override Prices GetPrices(Security asset, OrderDirection direction)
        {
            var lastData = asset.GetLastData();

            if (lastData == null)
            {
                return base.GetPrices(asset, direction);
            }

            try
            {
                var dynamicData = lastData as DynamicData;

                if (dynamicData != null)
                {
                    var current = asset.Price;
                    var endTime = lastData.EndTime;

                    // DynamicData.GetProperty() lowercases keys internally,
                    // matching how SetProperty() stores them.
                    decimal open = current, high = current, low = current, close = current;

                    if (dynamicData.HasProperty("Open"))
                        open = Convert.ToDecimal(dynamicData.GetProperty("Open"));
                    if (dynamicData.HasProperty("High"))
                        high = Convert.ToDecimal(dynamicData.GetProperty("High"));
                    if (dynamicData.HasProperty("Low"))
                        low = Convert.ToDecimal(dynamicData.GetProperty("Low"));
                    if (dynamicData.HasProperty("Close"))
                        close = Convert.ToDecimal(dynamicData.GetProperty("Close"));

                    if (!_loggedOnce)
                    {
                        Log.Trace($"CustomDataFillModel: type={lastData.GetType().Name}, " +
                                  $"OHLC=[{open}, {high}, {low}, {close}], " +
                                  $"asset.High={asset.High}, asset.Low={asset.Low}");
                        _loggedOnce = true;
                    }

                    return new Prices(endTime, current, open, high, low, close);
                }
                else
                {
                    if (!_loggedOnce)
                    {
                        Log.Trace($"CustomDataFillModel: type={lastData.GetType().Name} " +
                                  $"is NOT DynamicData, falling back to base");
                        _loggedOnce = true;
                    }
                    return base.GetPrices(asset, direction);
                }
            }
            catch (Exception ex)
            {
                if (!_loggedOnce)
                {
                    Log.Trace($"CustomDataFillModel: Exception in GetPrices: {ex.Message}");
                    _loggedOnce = true;
                }
                return base.GetPrices(asset, direction);
            }
        }
    }
}
