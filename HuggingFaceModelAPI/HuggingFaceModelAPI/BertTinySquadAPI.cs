using System.Net;
using System.Text.Json;
using Microsoft.Azure.Functions.Worker;
using Microsoft.Azure.Functions.Worker.Http;
using Microsoft.Extensions.Logging;

namespace HuggingFaceModelAPI
{
    public class BertTinySquadAPI
    {
        private readonly ILogger _logger;

        public BertTinySquadAPI(ILoggerFactory loggerFactory)
        {
            _logger = loggerFactory.CreateLogger<BertTinySquadAPI>();
        }

        [Function("BertTinySquadAPI")]
        public async Task<HttpResponseData> Run([HttpTrigger(AuthorizationLevel.Function, "get", "post")] HttpRequestData req)
        {
            _logger.LogInformation("C# HTTP trigger function processed a request.");

           

            StreamReader reader = new StreamReader(req.Body);
            string inputjson = reader.ReadToEnd();

            var data = JsonSerializer.Deserialize<JsonElement>(inputjson);

            // Get the question and context keys from the JSON object
            string question = data.GetProperty("question").GetString()??"";
            string context = data.GetProperty("context").GetString()??"";

            // Do something with the question and context, such as calling a model API
            // For example, using the Hugging Face Inference API
            string modelId = "VenkatManda/bert-venkat-test";
            string apiToken = Environment.GetEnvironmentVariable("apiToken")??""; // get yours at hf.co/settings/tokens

            
            var client = new HttpClient();
            client.DefaultRequestHeaders.Authorization = new System.Net.Http.Headers.AuthenticationHeaderValue("Bearer", apiToken);

            // Create the payload as a JSON object
            var payload = new
            {
                inputs = new { question = question, context = context }
            };

           
            var json = JsonSerializer.Serialize(payload);

          
            var response = await client.PostAsync($"https://api-inference.huggingface.co/models/{modelId}", new StringContent(json));

            
            var result = await response.Content.ReadAsStringAsync();

            var actualResponse = req.CreateResponse(response.StatusCode);
            var output = await response.Content.ReadAsStringAsync();
            actualResponse.WriteString(output);


            return actualResponse;
        }
    }
}
