/*
 * LEAN Data Queue Handler for GCP Pub/Sub
 * 
 * This implements IDataQueueHandler to subscribe to Pub/Sub topics
 * and feed real-time data to LEAN's engine using REST API.
 */

using System;
using System.Collections.Generic;
using System.ComponentModel.Composition;
using System.Linq;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using QuantConnect.Data;
using QuantConnect.Data.Market;
using QuantConnect.Interfaces;
using QuantConnect.Logging;
using QuantConnect.Packets;
using QuantConnect.Util;
using QuantConnect.Lean.Engine.DataFeeds.Enumerators;
using Newtonsoft.Json;
using Newtonsoft.Json.Linq;

namespace QuantConnect.Lean.Engine.DataFeeds
{
    /// <summary>
    /// Data queue handler for GCP Pub/Sub using REST API
    /// </summary>
    [Export(typeof(IDataQueueHandler))]
    public class PubSubDataQueueHandler : IDataQueueHandler
    {
        private string _projectId;
        private Dictionary<Symbol, SubscriptionDataConfig> _subscriptions;
        private Dictionary<Symbol, Task> _subscriptionTasks;
        private Dictionary<Symbol, EnqueueableEnumerator<BaseData>> _enumerators;
        private CancellationTokenSource _cancellationTokenSource;
        private readonly object _lock = new object();
        private bool _isConnected;
        
        // Global event handler for notifying LEAN of new data
        // LEAN's LiveSubscriptionEnumerator uses this to know when to call MoveNext()
        private EventHandler _globalDataAvailableHandler;
        
        private HttpClient _restApiClient;
        private string _accessToken;
        private DateTime _tokenExpiry;
        private ServiceAccountCredentialData _credentialData;

        public bool IsConnected
        {
            get
            {
                if (string.IsNullOrEmpty(_projectId))
                    return false;
                
                var credentialsPath = Environment.GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS");
                if (string.IsNullOrEmpty(credentialsPath) || !System.IO.File.Exists(credentialsPath))
                    return false;
                
                return _isConnected && _restApiClient != null;
            }
        }

        public PubSubDataQueueHandler()
        {
            AppContext.SetSwitch("System.Net.Http.UseSocketsHttpHandler", true);
            
            Log.Trace("PubSubDataQueueHandler: Initializing...");
            
            _projectId = Environment.GetEnvironmentVariable("GOOGLE_CLOUD_PROJECT") ?? "";
            _subscriptions = new Dictionary<Symbol, SubscriptionDataConfig>();
            _subscriptionTasks = new Dictionary<Symbol, Task>();
            _enumerators = new Dictionary<Symbol, EnqueueableEnumerator<BaseData>>();
            _cancellationTokenSource = new CancellationTokenSource();
            _isConnected = false;

            if (string.IsNullOrEmpty(_projectId))
            {
                Log.Error("PubSubDataQueueHandler: GOOGLE_CLOUD_PROJECT environment variable is required");
                return;
            }

            var credentialsPath = Environment.GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS");
            if (string.IsNullOrEmpty(credentialsPath))
            {
                Log.Error("PubSubDataQueueHandler: GOOGLE_APPLICATION_CREDENTIALS environment variable is required");
                return;
            }

            if (!System.IO.File.Exists(credentialsPath))
            {
                Log.Error($"PubSubDataQueueHandler: Credentials file not found: {credentialsPath}");
                return;
            }
        }


        public IEnumerator<BaseData> Subscribe(SubscriptionDataConfig dataConfig, EventHandler newDataAvailableHandler)
        {
            if (string.IsNullOrEmpty(_projectId))
            {
                Log.Error("PubSubDataQueueHandler: Cannot subscribe - GOOGLE_CLOUD_PROJECT not set");
                return Enumerable.Empty<BaseData>().GetEnumerator();
            }

            var credentialsPath = Environment.GetEnvironmentVariable("GOOGLE_APPLICATION_CREDENTIALS");
            if (string.IsNullOrEmpty(credentialsPath) || !System.IO.File.Exists(credentialsPath))
            {
                Log.Error("PubSubDataQueueHandler: Cannot subscribe - GOOGLE_APPLICATION_CREDENTIALS not set or file not found");
                return Enumerable.Empty<BaseData>().GetEnumerator();
            }

            if (!_isConnected || _restApiClient == null)
            {
                try
                {
                    _isConnected = InitializeRestApiClient(credentialsPath);
                    if (!_isConnected)
                    {
                        Log.Error("PubSubDataQueueHandler: Failed to initialize REST API client");
                        return Enumerable.Empty<BaseData>().GetEnumerator();
                    }
                    Log.Trace($"PubSubDataQueueHandler: Connected (project: {_projectId})");
                }
                catch (Exception ex)
                {
                    Log.Error($"PubSubDataQueueHandler: Initialization failed: {ex.Message}");
                    _isConnected = false;
                    return Enumerable.Empty<BaseData>().GetEnumerator();
                }
            }

            try
            {
                var symbol = dataConfig.Symbol;
                
                // Skip internal/benchmark subscriptions - they shouldn't use Pub/Sub
                // Internal subscriptions are for LEAN's internal use (benchmarks, universe selection, etc.)
                if (dataConfig.IsInternalFeed || 
                    symbol.Value.Contains("QC-UNIVERSE") || 
                    (symbol.Value.StartsWith("SPY") && (symbol.Value.Contains("2T") || symbol.Value == "SPY")))
                {
                    Log.Trace($"PubSubDataQueueHandler: Skipping internal/benchmark subscription: {symbol} (IsInternal: {dataConfig.IsInternalFeed})");
                    return Enumerable.Empty<BaseData>().GetEnumerator();
                }
                
                // Store the global event handler (LEAN may pass the same handler for all subscriptions)
                // This ensures we can notify LEAN even if called from background threads
                lock (_lock)
                {
                    if (_globalDataAvailableHandler == null && newDataAvailableHandler != null)
                    {
                        _globalDataAvailableHandler = newDataAvailableHandler;
                    }
                }
                
                var testSubscription = Environment.GetEnvironmentVariable("PUBSUB_TEST_SUBSCRIPTION");
                
                string topicName = GetTopicName(symbol.Value);
                string subscriptionName = !string.IsNullOrEmpty(testSubscription) ? testSubscription : GetSubscriptionName(topicName);
                
                lock (_lock)
                {
                    _subscriptions[symbol] = dataConfig;
                }

                var subscriptionPath = $"projects/{_projectId}/subscriptions/{subscriptionName}";
                var topicPath = $"projects/{_projectId}/topics/{topicName}";
                
                var task = Task.Run(async () => await SubscribeViaRestApi(
                    symbol, subscriptionPath, topicPath, subscriptionName, topicName, 
                    dataConfig, newDataAvailableHandler, _cancellationTokenSource.Token));
                
                lock (_lock)
                {
                    _subscriptionTasks[symbol] = task;
                }

                // Create an EnqueueableEnumerator for this symbol
                // This is the pattern LEAN uses - we enqueue data when it arrives,
                // and LEAN's LiveSubscriptionEnumerator calls MoveNext() to retrieve it
                var enumerator = new EnqueueableEnumerator<BaseData>(blocking: false);
                lock (_lock)
                {
                    _enumerators[symbol] = enumerator;
                }
                
                Log.Trace($"PubSubDataQueueHandler: Subscribed to {symbol} via {subscriptionName}");
                
                return enumerator;
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Failed to subscribe to {dataConfig.Symbol}: {ex.Message}");
                return Enumerable.Empty<BaseData>().GetEnumerator();
            }
        }


        public void Unsubscribe(SubscriptionDataConfig dataConfig)
        {
            var symbol = dataConfig.Symbol;
            
            lock (_lock)
            {
                _subscriptions.Remove(symbol);
                _subscriptionTasks.Remove(symbol);
                
                // Stop and dispose the enumerator for this symbol
                if (_enumerators.TryGetValue(symbol, out var enumerator))
                {
                    enumerator.Stop();
                    enumerator.DisposeSafely();
                    _enumerators.Remove(symbol);
                }
            }
        }

        public void SetJob(LiveNodePacket job)
        {
            // No-op: job information not needed for Pub/Sub
        }

        public void Dispose()
        {
            _cancellationTokenSource?.Cancel();
            
            lock (_lock)
            {
                // Stop and dispose all enumerators
                foreach (var enumerator in _enumerators.Values)
                {
                    enumerator.Stop();
                    enumerator.DisposeSafely();
                }
                
                _subscriptions.Clear();
                _subscriptionTasks.Clear();
                _enumerators.Clear();
            }

            _restApiClient?.Dispose();
            _restApiClient = null;
            _isConnected = false;
        }

        private bool InitializeRestApiClient(string credentialsPath)
        {
            try
            {
                var handler = new SocketsHttpHandler();
                _restApiClient = new HttpClient(handler)
                {
                    BaseAddress = new Uri("https://pubsub.googleapis.com/v1/")
                };
                
                var credJson = System.IO.File.ReadAllText(credentialsPath);
                var credData = JsonConvert.DeserializeObject<ServiceAccountCredentialData>(credJson);
                
                if (credData == null || string.IsNullOrEmpty(credData.PrivateKey))
                {
                    Log.Error("PubSubDataQueueHandler: Invalid service account credentials");
                    return false;
                }
                
                _credentialData = credData;
                _accessToken = GetAccessToken(credData).GetAwaiter().GetResult();
                
                if (string.IsNullOrEmpty(_accessToken))
                {
                    Log.Error("PubSubDataQueueHandler: Failed to get access token");
                    return false;
                }
                
                return true;
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Initialization failed: {ex.Message}");
                return false;
            }
        }
        
        private async Task<string> GetAccessToken(ServiceAccountCredentialData credData)
        {
            try
            {
                if (!string.IsNullOrEmpty(_accessToken) && _tokenExpiry > DateTime.UtcNow.AddMinutes(5))
                {
                    return _accessToken;
                }
                
                var jwt = CreateJwtToken(credData);
                if (string.IsNullOrEmpty(jwt))
                {
                    Log.Error("PubSubDataQueueHandler: Failed to create JWT token");
                    return null;
                }
                
                var tokenResponse = await ExchangeJwtForAccessToken(jwt);
                if (tokenResponse != null && tokenResponse.ContainsKey("access_token"))
                {
                    _accessToken = tokenResponse["access_token"].ToString();
                    var expiresIn = tokenResponse.ContainsKey("expires_in") 
                        ? int.Parse(tokenResponse["expires_in"].ToString()) 
                        : 3600;
                    _tokenExpiry = DateTime.UtcNow.AddSeconds(expiresIn);
                    return _accessToken;
                }
                
                Log.Error("PubSubDataQueueHandler: Failed to exchange JWT for access token");
                return null;
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Failed to get access token: {ex.Message}");
                return null;
            }
        }
        
        private string CreateJwtToken(ServiceAccountCredentialData credData)
        {
            try
            {
                var now = DateTime.UtcNow;
                var header = new { alg = "RS256", typ = "JWT" };
                var claims = new
                {
                    iss = credData.ClientEmail,
                    scope = "https://www.googleapis.com/auth/pubsub",
                    aud = "https://oauth2.googleapis.com/token",
                    exp = ((DateTimeOffset)now.AddHours(1)).ToUnixTimeSeconds(),
                    iat = ((DateTimeOffset)now).ToUnixTimeSeconds()
                };
                
                var headerBase64 = Base64UrlEncode(Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(header)));
                var claimsBase64 = Base64UrlEncode(Encoding.UTF8.GetBytes(JsonConvert.SerializeObject(claims)));
                var unsignedToken = $"{headerBase64}.{claimsBase64}";
                
                var privateKey = credData.PrivateKey
                    .Replace("-----BEGIN PRIVATE KEY-----", "")
                    .Replace("-----END PRIVATE KEY-----", "")
                    .Replace("\n", "")
                    .Replace("\r", "")
                    .Replace(" ", "");
                
                using (var rsa = RSA.Create())
                {
                    rsa.ImportPkcs8PrivateKey(Convert.FromBase64String(privateKey), out _);
                    var signature = rsa.SignData(Encoding.UTF8.GetBytes(unsignedToken), HashAlgorithmName.SHA256, RSASignaturePadding.Pkcs1);
                    var signatureBase64 = Base64UrlEncode(signature);
                    return $"{unsignedToken}.{signatureBase64}";
                }
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Failed to create JWT: {ex.Message}");
                return null;
            }
        }
        
        private string Base64UrlEncode(byte[] input)
        {
            var base64 = Convert.ToBase64String(input);
            return base64.TrimEnd('=').Replace('+', '-').Replace('/', '_');
        }
        
        private async Task<Dictionary<string, object>> ExchangeJwtForAccessToken(string jwt)
        {
            try
            {
                var tokenRequest = new
                {
                    grant_type = "urn:ietf:params:oauth:grant-type:jwt-bearer",
                    assertion = jwt
                };
                
                var request = new HttpRequestMessage(HttpMethod.Post, "https://oauth2.googleapis.com/token")
                {
                    Content = new StringContent(JsonConvert.SerializeObject(tokenRequest), Encoding.UTF8, "application/json")
                };
                
                var response = await _restApiClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    return JsonConvert.DeserializeObject<Dictionary<string, object>>(content);
                }
                
                var errorContent = await response.Content.ReadAsStringAsync();
                Log.Error($"PubSubDataQueueHandler: Failed to exchange JWT: {response.StatusCode} - {errorContent}");
                return null;
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Error exchanging JWT: {ex.Message}");
                return null;
            }
        }

        private string GetTopicName(string symbol)
        {
            var normalized = symbol;
            if (!normalized.Contains("-") && normalized.Length >= 6)
            {
                var baseCurrency = normalized.Substring(0, 3);
                var quoteCurrency = normalized.Substring(3);
                normalized = $"{baseCurrency}-{quoteCurrency}";
            }
            return $"vibe-trade-candles-{normalized.ToLower()}-1m";
        }

        private string GetSubscriptionName(string topicName)
        {
            var parts = topicName.Split('-');
            if (parts.Length >= 4)
            {
                var symbolPart = string.Join("-", parts.Skip(2).Take(parts.Length - 3));
                return $"vibe-trade-lean-{symbolPart}-1m";
            }
            return $"vibe-trade-lean-{topicName}";
        }

        private async Task SubscribeViaRestApi(Symbol symbol, string subscriptionPath, string topicPath, 
            string subscriptionName, string topicName, SubscriptionDataConfig config, 
            EventHandler newDataAvailableHandler, CancellationToken cancellationToken)
        {
            try
            {
                // Try to verify subscription exists (non-blocking)
                try
                {
                    var subscription = await GetSubscriptionViaRestApi(subscriptionPath);
                    if (subscription == null && subscriptionName != "test_local")
                    {
                        await CreateSubscriptionViaRestApi(subscriptionPath, topicPath);
                    }
                }
                catch
                {
                    // Continue even if we can't verify subscription
                }
                
                while (!cancellationToken.IsCancellationRequested)
                {
                    try
                    {
                        var messages = await PullMessagesViaRestApi(subscriptionPath, maxMessages: 10);
                        
                        if (messages != null && messages.Count > 0)
                        {
                            var ackIds = new List<string>();
                            
                            foreach (var message in messages)
                            {
                                try
                                {
                                    if (!message.ContainsKey("message"))
                                        continue;
                                    
                                    Dictionary<string, object> messageData;
                                    if (message["message"] is JObject jObj)
                                    {
                                        messageData = jObj.ToObject<Dictionary<string, object>>();
                                    }
                                    else
                                    {
                                        messageData = message["message"] as Dictionary<string, object>;
                                    }
                                    
                                    if (messageData == null || !messageData.ContainsKey("data"))
                                        continue;
                                    
                                    var dataBytes = Convert.FromBase64String(messageData["data"].ToString());
                                    var data = ParseMessage(dataBytes, symbol, config);
                                    
                                    if (data != null)
                                    {
                                        // Enqueue data into the EnqueueableEnumerator for this symbol
                                        // LEAN's LiveSubscriptionEnumerator will call MoveNext() to retrieve it
                                        EnqueueableEnumerator<BaseData> enumerator = null;
                                        lock (_lock)
                                        {
                                            if (_enumerators.TryGetValue(symbol, out enumerator))
                                            {
                                                enumerator.Enqueue(data);
                                            }
                                            else
                                            {
                                                Log.Error($"PubSubDataQueueHandler: No enumerator found for {symbol} - data will be lost");
                                            }
                                        }
                                        
                                        // Notify LEAN that new data is available
                                        // This triggers LiveSubscriptionEnumerator to call MoveNext()
                                        EventHandler handlerToInvoke = null;
                                        lock (_lock)
                                        {
                                            handlerToInvoke = _globalDataAvailableHandler ?? newDataAvailableHandler;
                                        }
                                        
                                        if (handlerToInvoke != null)
                                        {
                                            try
                                            {
                                                handlerToInvoke.Invoke(this, EventArgs.Empty);
                                            }
                                            catch (Exception ex)
                                            {
                                                Log.Error($"PubSubDataQueueHandler: Error invoking data available handler: {ex.Message}");
                                            }
                                        }
                                    }
                                    
                                    if (message.ContainsKey("ackId"))
                                    {
                                        ackIds.Add(message["ackId"].ToString());
                                    }
                                }
                                catch (Exception msgEx)
                                {
                                    Log.Error($"PubSubDataQueueHandler: Error processing message: {msgEx.Message}");
                                }
                            }
                            
                            if (ackIds.Count > 0)
                            {
                                await AcknowledgeMessagesViaRestApi(subscriptionPath, ackIds);
                            }
                        }
                        
                        await Task.Delay(100, cancellationToken);
                    }
                    catch (Exception pullEx)
                    {
                        Log.Error($"PubSubDataQueueHandler: Error in pull loop: {pullEx.Message}");
                        await Task.Delay(1000, cancellationToken);
                    }
                }
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Subscription error for {symbol}: {ex.Message}");
            }
        }
        
        private async Task<Dictionary<string, object>> GetSubscriptionViaRestApi(string subscriptionPath)
        {
            try
            {
                await EnsureAccessToken();
                
                var request = new HttpRequestMessage(HttpMethod.Get, subscriptionPath);
                request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _accessToken);
                
                var response = await _restApiClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    return JsonConvert.DeserializeObject<Dictionary<string, object>>(content);
                }
                
                return null;
            }
            catch
            {
                return null;
            }
        }
        
        private async Task<bool> CreateSubscriptionViaRestApi(string subscriptionPath, string topicPath)
        {
            try
            {
                await EnsureAccessToken();
                
                var subscriptionBody = new
                {
                    topic = topicPath,
                    ackDeadlineSeconds = 60
                };
                
                var request = new HttpRequestMessage(HttpMethod.Put, subscriptionPath)
                {
                    Content = new StringContent(JsonConvert.SerializeObject(subscriptionBody), Encoding.UTF8, "application/json")
                };
                request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _accessToken);
                
                var response = await _restApiClient.SendAsync(request);
                return response.IsSuccessStatusCode;
            }
            catch
            {
                return false;
            }
        }
        
        private async Task<List<Dictionary<string, object>>> PullMessagesViaRestApi(string subscriptionPath, int maxMessages = 10)
        {
            try
            {
                await EnsureAccessToken();
                
                var pullBody = new
                {
                    maxMessages = maxMessages,
                    returnImmediately = false
                };
                
                var request = new HttpRequestMessage(HttpMethod.Post, $"{subscriptionPath}:pull")
                {
                    Content = new StringContent(JsonConvert.SerializeObject(pullBody), Encoding.UTF8, "application/json")
                };
                request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _accessToken);
                
                var response = await _restApiClient.SendAsync(request);
                if (response.IsSuccessStatusCode)
                {
                    var content = await response.Content.ReadAsStringAsync();
                    var result = JsonConvert.DeserializeObject<Dictionary<string, object>>(content);
                    
                    if (result != null && result.ContainsKey("receivedMessages"))
                    {
                        var messages = result["receivedMessages"] as JArray;
                        if (messages != null)
                        {
                            return messages.ToObject<List<Dictionary<string, object>>>();
                        }
                    }
                }
                else
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    Log.Error($"PubSubDataQueueHandler: Failed to pull messages: {response.StatusCode} - {errorContent}");
                }
                
                return new List<Dictionary<string, object>>();
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Error pulling messages: {ex.Message}");
                return new List<Dictionary<string, object>>();
            }
        }
        
        private async Task AcknowledgeMessagesViaRestApi(string subscriptionPath, List<string> ackIds)
        {
            try
            {
                await EnsureAccessToken();
                
                var ackBody = new { ackIds = ackIds };
                var request = new HttpRequestMessage(HttpMethod.Post, $"{subscriptionPath}:acknowledge")
                {
                    Content = new StringContent(JsonConvert.SerializeObject(ackBody), Encoding.UTF8, "application/json")
                };
                request.Headers.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", _accessToken);
                
                var response = await _restApiClient.SendAsync(request);
                if (!response.IsSuccessStatusCode)
                {
                    var errorContent = await response.Content.ReadAsStringAsync();
                    Log.Error($"PubSubDataQueueHandler: Failed to acknowledge messages: {response.StatusCode} - {errorContent}");
                }
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Error acknowledging messages: {ex.Message}");
            }
        }
        
        private async Task EnsureAccessToken()
        {
            if (string.IsNullOrEmpty(_accessToken) || _tokenExpiry <= DateTime.UtcNow.AddMinutes(5))
            {
                _accessToken = await GetAccessToken(_credentialData);
                if (string.IsNullOrEmpty(_accessToken))
                {
                    throw new Exception("Failed to get access token");
                }
            }
        }

        private BaseData ParseMessage(byte[] messageData, Symbol symbol, SubscriptionDataConfig config)
        {
            try
            {
                var json = Encoding.UTF8.GetString(messageData);
                var data = JsonConvert.DeserializeObject<PubSubMessage>(json);

                if (data == null)
                {
                    Log.Error("PubSubDataQueueHandler: Failed to deserialize message");
                    return null;
                }

                DateTime timestamp;
                try
                {
                    timestamp = DateTime.Parse(data.Timestamp);
                    if (data.Timestamp.EndsWith("Z"))
                    {
                        timestamp = timestamp.ToUniversalTime();
                    }
                }
                catch (Exception ex)
                {
                    Log.Error($"PubSubDataQueueHandler: Failed to parse timestamp '{data.Timestamp}': {ex.Message}");
                    return null;
                }

                if (data.Open <= 0 || data.High <= 0 || data.Low <= 0 || data.Close <= 0)
                {
                    Log.Error($"PubSubDataQueueHandler: Invalid price data");
                    return null;
                }

                // Use the symbol from LEAN config (what algorithm requested)
                // If algorithm uses BTC-USD, symbol will be BTC-USD - no mapping needed
                // The handler uses the symbol as-is, ensuring TradeBar matches the subscription
                
                // Ensure timestamp is in UTC and not too far in the future
                // LEAN filters out data that's too far ahead of algorithm time
                if (timestamp.Kind != DateTimeKind.Utc)
                {
                    timestamp = timestamp.ToUniversalTime();
                }
                
                // In live mode, LEAN expects data to be at or slightly before current time
                // If data is too far in the future, adjust it to current time
                var now = DateTime.UtcNow;
                if (timestamp > now.AddMinutes(5))
                {
                    Log.Trace($"PubSubDataQueueHandler: Data timestamp {timestamp} is too far in future (now={now}), adjusting to current time");
                    timestamp = now;
                }
                
                return new TradeBar(timestamp, symbol, data.Open, data.High, data.Low, data.Close, data.Volume);
            }
            catch (Exception ex)
            {
                Log.Error($"PubSubDataQueueHandler: Failed to parse message: {ex.Message}");
                return null;
            }
        }

        private class ServiceAccountCredentialData
        {
            [JsonProperty("type")]
            public string Type { get; set; }
            
            [JsonProperty("project_id")]
            public string ProjectId { get; set; }
            
            [JsonProperty("private_key_id")]
            public string PrivateKeyId { get; set; }
            
            [JsonProperty("private_key")]
            public string PrivateKey { get; set; }
            
            [JsonProperty("client_email")]
            public string ClientEmail { get; set; }
            
            [JsonProperty("client_id")]
            public string ClientId { get; set; }
            
            [JsonProperty("auth_uri")]
            public string AuthUri { get; set; }
            
            [JsonProperty("token_uri")]
            public string TokenUri { get; set; }
        }

        private class PubSubMessage
        {
            [JsonProperty("symbol")]
            public string Symbol { get; set; }

            [JsonProperty("timestamp")]
            public string Timestamp { get; set; }

            [JsonProperty("open")]
            public decimal Open { get; set; }

            [JsonProperty("high")]
            public decimal High { get; set; }

            [JsonProperty("low")]
            public decimal Low { get; set; }

            [JsonProperty("close")]
            public decimal Close { get; set; }

            [JsonProperty("volume")]
            public decimal Volume { get; set; }

            [JsonProperty("granularity")]
            public string Granularity { get; set; }
        }
    }
}
