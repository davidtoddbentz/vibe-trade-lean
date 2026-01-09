"""
Test strategy algorithm that actually trades to verify Pub/Sub data flow and execution.
This algorithm places orders based on incoming data to verify end-to-end functionality.
"""

from AlgorithmImports import *
import os

class PubSubStrategyAlgorithm(QCAlgorithm):
    """Strategy algorithm that trades to verify Pub/Sub integration works correctly."""
    
    def Initialize(self):
        """Initialize the algorithm."""
        # In live mode, don't set dates - LEAN uses current time
        # Setting dates can cause the algorithm clock to start in the past,
        # making data timestamps appear in the future
        if not self.LiveMode:
            self.SetStartDate(2024, 1, 1)
            self.SetEndDate(2024, 12, 31)
        self.SetCash(100000)
        
        # Get symbol from environment variable (default: BTC-USD)
        symbol_str = os.getenv("PUBSUB_TEST_SYMBOL", "BTC-USD")
        
        # Create symbol properties manually
        asset_name = "Bitcoin" if "BTC" in symbol_str else "Ethereum" if "ETH" in symbol_str else symbol_str.split("-")[0]
        properties = SymbolProperties(asset_name, "USD", 1, 0.01, 0.01, symbol_str)
        exchangeHours = SecurityExchangeHours.AlwaysOpen(TimeZones.Utc)
        
        # Use AddData to bypass symbol database
        self.symbol = self.AddData(
            TradeBar,
            symbol_str,
            properties,
            exchangeHours,
            Resolution.Minute
        ).Symbol
        
        self.Log(f"✅ Strategy initialized with symbol: {symbol_str}")
        self.Log(f"⏰ Algorithm clock time: {self.Time}")
        self.Log(f"🔄 Live mode: {self.LiveMode}")
        
        # Track metrics for verification
        self.data_points_received = 0
        self.orders_placed = 0
        self.orders_filled = 0
        self.last_price = None
        self.first_data_time = None
        self.last_data_time = None
        
        # Simple strategy: Buy on first data point, sell after 5 bars
        self.bars_since_entry = 0
        self.entry_price = None
        self.position_entered = False
    
    def OnData(self, data):
        """Called when new data arrives from Pub/Sub."""
        self.Log(f"🔔 OnData called at {self.Time} with {len(data)} symbols")
        if self.symbol in data and data[self.symbol] is not None:
            bar = data[self.symbol]
            self.data_points_received += 1
            self.last_data_time = bar.Time
            self.last_price = bar.Close
            
            if self.first_data_time is None:
                self.first_data_time = bar.Time
                self.Log(f"📊 First data received: {bar.Time} | Price: ${bar.Close:.2f}")
            
            # Strategy: Buy on first bar, sell after 3 bars (faster for testing)
            if not self.position_entered and self.data_points_received == 1:
                # Enter position: 10% of portfolio
                quantity = self.CalculateOrderQuantity(self.symbol, 0.1)
                order = self.MarketOrder(self.symbol, quantity)
                self.orders_placed += 1
                self.entry_price = bar.Close
                self.position_entered = True
                self.Log(f"🛒 BUY Order #{self.orders_placed}: {quantity} @ ${bar.Close:.2f}")
            
            elif self.position_entered:
                self.bars_since_entry += 1
                
                # Exit after 3 bars (reduced from 5 for faster testing)
                if self.bars_since_entry >= 3:
                    if self.Portfolio[self.symbol].Invested:
                        self.Liquidate(self.symbol)
                        self.orders_placed += 1
                        self.Log(f"🛒 SELL Order #{self.orders_placed}: Liquidate @ ${bar.Close:.2f}")
                        self.position_entered = False
                        self.bars_since_entry = 0
            
            # Log every 10th data point
            if self.data_points_received % 10 == 0:
                self.Log(f"📈 Data #{self.data_points_received}: {bar.Time} | Close: ${bar.Close:.2f} | Position: {self.Portfolio[self.symbol].Quantity}")
    
    def OnOrderEvent(self, orderEvent):
        """Called when order status changes."""
        if orderEvent.Status == OrderStatus.Filled:
            self.orders_filled += 1
            direction = "BUY" if orderEvent.FillQuantity > 0 else "SELL"
            self.Log(f"✅ Order Filled #{self.orders_filled}: {direction} {abs(orderEvent.FillQuantity)} @ ${orderEvent.FillPrice:.2f}")
        elif orderEvent.Status == OrderStatus.Submitted:
            self.Log(f"📤 Order Submitted: {orderEvent.Symbol} {orderEvent.Quantity}")
        elif orderEvent.Status == OrderStatus.Invalid:
            self.Log(f"❌ Order Invalid: {orderEvent.Message}")
    
    def OnEndOfAlgorithm(self):
        """Called when algorithm ends."""
        self.Log(f"🏁 Algorithm ended")
        self.Log(f"📊 Total data points: {self.data_points_received}")
        self.Log(f"🛒 Orders placed: {self.orders_placed}")
        self.Log(f"✅ Orders filled: {self.orders_filled}")
        if self.first_data_time:
            self.Log(f"⏰ First data: {self.first_data_time}")
        if self.last_data_time:
            self.Log(f"⏰ Last data: {self.last_data_time}")
        
        # Log final portfolio state
        if self.Portfolio[self.symbol].Invested:
            self.Log(f"💼 Final position: {self.Portfolio[self.symbol].Quantity} @ ${self.Portfolio[self.symbol].Price:.2f}")
        else:
            self.Log(f"💼 Final position: FLAT")

