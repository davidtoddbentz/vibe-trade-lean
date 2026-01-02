"""
Test algorithm to verify PubSubDataQueueHandler works.
This algorithm subscribes to BTC-USD data from Pub/Sub and logs received data.
"""

from AlgorithmImports import *

class PubSubTestAlgorithm(QCAlgorithm):
    """Simple test algorithm to verify Pub/Sub data queue handler."""
    
    def Initialize(self):
        """Initialize the algorithm."""
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Try using BTC-USD directly - no mapping needed if symbol database works
        # Or use AddData with BTC-USD directly
        try:
            # First try: Use BTC-USD directly with AddData
            properties = SymbolProperties("Bitcoin", "USD", 1, 0.01, 0.01, "BTC-USD")
            exchangeHours = SecurityExchangeHours.AlwaysOpen(TimeZones.Utc)
            
            self.symbol = self.AddData(
                TradeBar,
                "BTC-USD",  # Use BTC-USD directly - no mapping needed
                properties,
                exchangeHours,
                Resolution.Minute
            ).Symbol
            self.Log(f"✅ Added BTC-USD symbol directly (no mapping needed)")
        except Exception as e:
            self.Log(f"❌ Failed to add BTC-USD: {e}")
            # Fallback to BTCUSD if BTC-USD doesn't work
            properties = SymbolProperties("Bitcoin", "USD", 1, 0.01, 0.01, "BTCUSD")
            exchangeHours = SecurityExchangeHours.AlwaysOpen(TimeZones.Utc)
            self.symbol = self.AddData(
                TradeBar,
                "BTCUSD",
                properties,
                exchangeHours,
                Resolution.Minute
            ).Symbol
            self.Log(f"✅ Added BTCUSD symbol (fallback)")
        
        self.Log(f"✅ Algorithm initialized")
        self.Log(f"📊 Symbol: {self.symbol}")
        self.Log(f"🔍 Waiting for data from Pub/Sub...")
        
        # Track data received
        self.data_count = 0
        self.last_data_time = None
    
    def OnData(self, data):
        """Called when new data arrives from Pub/Sub."""
        if self.symbol in data and data[self.symbol] is not None:
            bar = data[self.symbol]
            self.data_count += 1
            self.last_data_time = bar.Time
            
            # Log every 10th data point to avoid spam
            if self.data_count % 10 == 0:
                self.Log(f"📈 Received data #{self.data_count}: {bar.Time} | Close: ${bar.Close:.2f} | Volume: {bar.Volume:.2f}")
            
            # Check if we're in live mode
            if self.LiveMode:
                self.Log(f"🟢 LIVE MODE: Data received from Pub/Sub!")
            else:
                self.Log(f"🟡 BACKTEST MODE: Using file data")
    
    def OnEndOfAlgorithm(self):
        """Called when algorithm ends."""
        self.Log(f"🏁 Algorithm ended")
        self.Log(f"📊 Total data points received: {self.data_count}")
        if self.last_data_time:
            self.Log(f"⏰ Last data time: {self.last_data_time}")

