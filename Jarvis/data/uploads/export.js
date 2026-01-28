const fs = require("fs");
const { createClient } = require("@supabase/supabase-js");
const { unparse } = require("papaparse");

// Load environment variables from GitHub Actions
const supabaseUrl = process.env.SUPABASE_URL;
const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
const csvFileName = process.env.CSV_FILENAME || "dataset.csv";

// Initialize Supabase client
const supabase = createClient(supabaseUrl, supabaseKey);

// Define all table names to export
const tableNames = [
  'general_information',
  'birth_history',
  'educational_history',
  'medical_history',
  'sensory_analysis',
  'psychological_analysis',
  'occupational_therapy',
  'social_and_emotional_history',
  'speech_and_language_history',
  'speech_and_language_assessment',
  'skill_recommendations',
  // Individual milestone tables
  'stand',
  'sit',
  'neck_holding',
  'crawl',
  'walk',
  'use_toilet',
  'use_of_combine_words',
  'use_of_single_words',
  // Individual medical condition tables
  'surgery',
  'vision',
  'allergy',
  'illnesses',
  'infection'
];

async function exportAllTablesAsCSV() {
  try {
    console.log(`🚀 Starting export and joining data from ${tableNames.length} tables...`);
    
    // Start with general_information as the base table
    const { data: baseData, error: baseError } = await supabase
      .from('general_information')
      .select("*");

    if (baseError || !baseData || baseData.length === 0) {
      console.error("❌ Error fetching base data from general_information:", baseError?.message);
      process.exit(1);
    }

    console.log(`✅ Base data: ${baseData.length} rows from general_information`);
    
    // Create a map with id as key for easy joining
    const combinedData = {};
    baseData.forEach(row => {
      combinedData[row.id] = { ...row };
    });

    // Join data from all other tables
    const otherTables = tableNames.filter(table => table !== 'general_information');
    
    for (const tableName of otherTables) {
      console.log(`📊 Fetching and joining data from ${tableName}...`);
      
      const { data, error } = await supabase
        .from(tableName)
        .select("*");

      if (error) {
        console.warn(`⚠️ Error fetching from ${tableName}:`, error.message);
        continue;
      }

      if (data && data.length > 0) {
        // Join data based on id
        data.forEach(row => {
          const id = row.id;
          if (combinedData[id]) {
            // Merge the row data, prefixing column names with table name to avoid conflicts
            Object.keys(row).forEach(key => {
              if (key !== 'id') { // Don't duplicate the id field
                const prefixedKey = `${tableName}_${key}`;
                combinedData[id][prefixedKey] = row[key];
              }
            });
          }
        });
        console.log(`✅ Joined ${data.length} rows from ${tableName}`);
      } else {
        console.log(`📝 No data found in ${tableName}`);
      }
    }

    // Convert to array
    const finalData = Object.values(combinedData);
    
    if (finalData.length === 0) {
      console.warn("⚠️ No combined data found.");
      fs.writeFileSync(csvFileName, "", "utf8");
      process.exit(0);
    }

    // Convert JSON to CSV
    const csv = unparse(finalData);

    // Write to file
    fs.writeFileSync(csvFileName, csv, "utf8");
    console.log(`🎉 Successfully exported ${finalData.length} combined records with data from ${tableNames.length} tables to ${csvFileName}`);
    
  } catch (err) {
    console.error("🚨 Unexpected error:", err);
    process.exit(1);
  }
}

exportAllTablesAsCSV();
