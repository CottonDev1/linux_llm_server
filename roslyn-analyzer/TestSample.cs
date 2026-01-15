using System;
using System.Data;
using System.Data.SqlClient;
using System.Collections.Generic;

namespace TestApplication
{
    // Main business logic class
    public class BaleProcessor
    {
        private readonly BaleDataLayer _dataLayer;
        private readonly ILogger _logger;

        public BaleProcessor(BaleDataLayer dataLayer, ILogger logger)
        {
            _dataLayer = dataLayer;
            _logger = logger;
        }

        // This method is called by UI layer
        public bool ProcessBale(int baleNumber, string producerCode)
        {
            try
            {
                _logger.LogInfo($"Processing bale {baleNumber}");

                // Validate bale
                if (!ValidateBale(baleNumber))
                {
                    return false;
                }

                // Get bale data from database
                var baleData = _dataLayer.GetBaleData(baleNumber);

                // Process the bale
                var result = PerformBaleProcessing(baleData, producerCode);

                // Save results
                _dataLayer.SaveBaleResults(baleNumber, result);

                // Call stored procedure
                ExecuteBaleCommit(baleNumber);

                return true;
            }
            catch (Exception ex)
            {
                _logger.LogError($"Error processing bale: {ex.Message}");
                return false;
            }
        }

        private bool ValidateBale(int baleNumber)
        {
            // Validation logic
            return baleNumber > 0;
        }

        private BaleResult PerformBaleProcessing(DataRow baleData, string producerCode)
        {
            // Complex processing logic
            var calculator = new BaleCalculator();
            return calculator.Calculate(baleData, producerCode);
        }

        private void ExecuteBaleCommit(int baleNumber)
        {
            using (var conn = new SqlConnection("Server=NCSQLTEST;Database=Gin;"))
            {
                using (var cmd = new SqlCommand("usp_CommitBale", conn))
                {
                    cmd.CommandType = CommandType.StoredProcedure;
                    cmd.Parameters.AddWithValue("@BaleNumber", baleNumber);
                    cmd.Parameters.AddWithValue("@CommitDate", DateTime.Now);

                    conn.Open();
                    cmd.ExecuteNonQuery();
                }
            }
        }
    }

    // Data access layer
    public class BaleDataLayer
    {
        private readonly string _connectionString;

        public BaleDataLayer(string connectionString)
        {
            _connectionString = connectionString;
        }

        public DataRow GetBaleData(int baleNumber)
        {
            using (var adapter = new SqlDataAdapter(
                $"SELECT * FROM Bales WHERE BaleNumber = {baleNumber}",
                _connectionString))
            {
                var dt = new DataTable();
                adapter.Fill(dt);
                return dt.Rows.Count > 0 ? dt.Rows[0] : null;
            }
        }

        public void SaveBaleResults(int baleNumber, BaleResult result)
        {
            using (var conn = new SqlConnection(_connectionString))
            {
                var sql = "UPDATE Bales SET ProcessedDate = @Date, Result = @Result WHERE BaleNumber = @BaleNumber";
                using (var cmd = new SqlCommand(sql, conn))
                {
                    cmd.Parameters.AddWithValue("@Date", DateTime.Now);
                    cmd.Parameters.AddWithValue("@Result", result.ToString());
                    cmd.Parameters.AddWithValue("@BaleNumber", baleNumber);

                    conn.Open();
                    cmd.ExecuteNonQuery();
                }
            }
        }

        public List<int> GetPendingBales()
        {
            var bales = new List<int>();
            ExecuteStoredProcedure("usp_GetPendingBales", null, reader =>
            {
                while (reader.Read())
                {
                    bales.Add(reader.GetInt32(0));
                }
            });
            return bales;
        }

        private void ExecuteStoredProcedure(string procName, SqlParameter[] parameters, Action<SqlDataReader> processResults)
        {
            using (var conn = new SqlConnection(_connectionString))
            using (var cmd = new SqlCommand(procName, conn))
            {
                cmd.CommandType = CommandType.StoredProcedure;
                if (parameters != null)
                {
                    cmd.Parameters.AddRange(parameters);
                }

                conn.Open();
                using (var reader = cmd.ExecuteReader())
                {
                    processResults?.Invoke(reader);
                }
            }
        }
    }

    // Supporting classes
    public class BaleCalculator
    {
        public BaleResult Calculate(DataRow baleData, string producerCode)
        {
            // Calculation logic
            var service = new CalculationService();
            return service.PerformCalculation(baleData, producerCode);
        }
    }

    public class CalculationService
    {
        public BaleResult PerformCalculation(DataRow data, string code)
        {
            return new BaleResult { Success = true };
        }
    }

    public class BaleResult
    {
        public bool Success { get; set; }
        public override string ToString() => Success ? "Success" : "Failed";
    }

    public interface ILogger
    {
        void LogInfo(string message);
        void LogError(string message);
    }

    // UI Layer
    public class BaleEntryForm
    {
        private readonly BaleProcessor _processor;

        public BaleEntryForm()
        {
            var dataLayer = new BaleDataLayer("Server=NCSQLTEST;Database=Gin;");
            var logger = new ConsoleLogger();
            _processor = new BaleProcessor(dataLayer, logger);
        }

        // This is called when user clicks Save button
        public void SaveButton_Click(object sender, EventArgs e)
        {
            int baleNumber = 12345;
            string producerCode = "PROD001";

            bool success = _processor.ProcessBale(baleNumber, producerCode);

            if (success)
            {
                ShowMessage("Bale processed successfully!");
            }
        }

        private void ShowMessage(string message)
        {
            Console.WriteLine(message);
        }
    }

    public class ConsoleLogger : ILogger
    {
        public void LogInfo(string message) => Console.WriteLine($"INFO: {message}");
        public void LogError(string message) => Console.WriteLine($"ERROR: {message}");
    }

    public class EventArgs { }
}