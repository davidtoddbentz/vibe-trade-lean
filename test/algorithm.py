"""
Test algorithm to verify PubSubDataQueueHandler works.
This algorithm subscribes to crypto data from Pub/Sub and logs received data.
Supports any symbol via environment variables.
"""

from AlgorithmImports import *
import os

class PubSubTestAlgorithm(QCAlgorithm):
    """Simple test algorithm to verify Pub/Sub data queue handler."""
    
    def Initialize(self):
        """Initialize the algorithm."""
        self.SetStartDate(2024, 1, 1)
        self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Get symbol from environment variable (default: BTC-USD)
        # Can be set via: PUBSUB_TEST_SYMBOL=ETH-USD
        symbol_str = os.getenv("PUBSUB_TEST_SYMBOL", "BTC-USD")
        
        # Create symbol properties manually
        # This bypasses the symbol-properties-database.csv requirement
        # Based on: https://www.quantconnect.com/forum/discussion/13389/
        
        asset_name = "Bitcoin" if "BTC" in symbol_str else "Ethereum" if "ETH" in symbol_str else symbol_str.split("-")[0]
        properties = SymbolProperties(asset_name, "USD", 1, 0.01, 0.01, symbol_str)
        exchangeHours = SecurityExchangeHours.AlwaysOpen(TimeZones.Utc)
        
        # Use AddData instead of AddCrypto to bypass symbol database
        # The data queue handler will automatically use the correct subscription
        # based on PUBSUB_SUBSCRIPTION_{SYMBOL} or PUBSUB_TEST_SUBSCRIPTION
        self.symbol = self.AddData(
            TradeBar,  # Use TradeBar as the data type
            symbol_str,
            properties,
            exchangeHours,
            Resolution.Minute
        ).Symbol
        
        self.Log(f"✅ Added {symbol_str} symbol directly (no mapping needed)")
        
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
