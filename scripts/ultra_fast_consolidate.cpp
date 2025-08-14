#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>
#include <filesystem>
#include <thread>
#include <future>
#include <chrono>
#include <random>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <sys/stat.h>

namespace fs = std::filesystem;

// Ultra-fast JSON parser (simplified for speed)
class FastJsonParser {
private:
    std::string buffer;
    size_t pos = 0;
    
public:
    bool parseObject(std::unordered_map<std::string, std::string>& result) {
        result.clear();
        if (pos >= buffer.length() || buffer[pos] != '{') return false;
        pos++; // skip {
        
        while (pos < buffer.length() && buffer[pos] != '}') {
            // Skip whitespace
            while (pos < buffer.length() && (buffer[pos] == ' ' || buffer[pos] == '\n' || buffer[pos] == '\r' || buffer[pos] == '\t')) pos++;
            if (pos >= buffer.length()) break;
            
            // Parse key
            if (buffer[pos] != '"') break;
            pos++; // skip "
            size_t keyStart = pos;
            while (pos < buffer.length() && buffer[pos] != '"') pos++;
            if (pos >= buffer.length()) break;
            std::string key = buffer.substr(keyStart, pos - keyStart);
            pos++; // skip "
            
            // Skip whitespace and colon
            while (pos < buffer.length() && (buffer[pos] == ' ' || buffer[pos] == ':' || buffer[pos] == '\n' || buffer[pos] == '\r' || buffer[pos] == '\t')) pos++;
            if (pos >= buffer.length()) break;
            
            // Parse value
            if (buffer[pos] != '"') break;
            pos++; // skip "
            size_t valueStart = pos;
            while (pos < buffer.length() && buffer[pos] != '"') pos++;
            if (pos >= buffer.length()) break;
            std::string value = buffer.substr(valueStart, pos - valueStart);
            pos++; // skip "
            
            result[key] = value;
            
            // Skip whitespace and comma
            while (pos < buffer.length() && (buffer[pos] == ' ' || buffer[pos] == ',' || buffer[pos] == '\n' || buffer[pos] == '\r' || buffer[pos] == '\t')) pos++;
        }
        
        if (pos < buffer.length() && buffer[pos] == '}') {
            pos++; // skip }
            return true;
        }
        return false;
    }
    
    void setBuffer(const std::string& data) {
        buffer = data;
        pos = 0;
    }
};

// Ultra-fast hash function
class FastHash {
private:
    std::unordered_set<std::string> seen_hashes;
    std::mutex hash_mutex;
    
public:
    std::string computeHash(const std::string& text) {
        // Simple but fast hash
        uint64_t hash = 0;
        for (char c : text) {
            hash = hash * 31 + c;
        }
        return std::to_string(hash);
    }
    
    bool isDuplicate(const std::string& hash) {
        std::lock_guard<std::mutex> lock(hash_mutex);
        if (seen_hashes.find(hash) != seen_hashes.end()) {
            return true;
        }
        seen_hashes.insert(hash);
        return false;
    }
};

// Ultra-fast file processor
class UltraFastConsolidator {
private:
    std::string data_dir;
    std::string output_dir;
    int max_workers;
    FastHash hash_processor;
    
    std::vector<std::string> priority_datasets = {
        "pile_rml_final",
        "consolidated_rml", 
        "rml_extracted",
        "streaming_rml_output",
        "extracted RML DATA",
        "rml_extraction_part2_fixed",
        "cpp_rml_output_v4",
        "cr_simple",
        "cr_production",
        "cpp_rml_output_v5",
        "real_redpajama",
        "continuous_rml_output",
        "commoncrawl",
        "c4_backup_20250731_040122",
        "RedPajama-Data"
    };
    
    std::vector<std::string> rml_components = {
        "concepts", "triples", "entities", "emotions", 
        "reasoning", "intents", "summaries", "events", "vectors"
    };
    
    // Thread-safe statistics
    std::atomic<int64_t> total_samples{0};
    std::atomic<int64_t> duplicates_removed{0};
    std::atomic<int64_t> files_processed{0};
    std::atomic<int64_t> total_size_mb{0};
    std::mutex stats_mutex;
    std::unordered_map<std::string, std::atomic<int64_t>> dataset_stats;
    
    // Output files
    std::ofstream train_file;
    std::ofstream val_file;
    std::ofstream test_file;
    std::mutex file_mutex;
    
    // Random number generator for splitting
    std::random_device rd;
    std::mt19937 gen{rd()};
    std::uniform_real_distribution<> dis{0.0, 1.0};
    
public:
    UltraFastConsolidator(const std::string& data_dir, const std::string& output_dir, int max_workers = 16)
        : data_dir(data_dir), output_dir(output_dir), max_workers(max_workers) {
        
        // Create output directories
        fs::create_directories(output_dir + "/train");
        fs::create_directories(output_dir + "/validation");
        fs::create_directories(output_dir + "/test");
        
        // Open output files
        train_file.open(output_dir + "/train/train.jsonl");
        val_file.open(output_dir + "/validation/validation.jsonl");
        test_file.open(output_dir + "/test/test.jsonl");
        
        // Initialize dataset stats
        for (const auto& dataset : priority_datasets) {
            dataset_stats[dataset] = 0;
        }
    }
    
    ~UltraFastConsolidator() {
        if (train_file.is_open()) train_file.close();
        if (val_file.is_open()) val_file.close();
        if (test_file.is_open()) test_file.close();
    }
    
    std::vector<std::pair<fs::path, std::string>> findRmlFiles() {
        std::vector<std::pair<fs::path, std::string>> rml_files;
        
        for (const auto& dataset_name : priority_datasets) {
            fs::path dataset_path = data_dir + "/" + dataset_name;
            if (!fs::exists(dataset_path)) {
                std::cout << "⚠️ Dataset not found: " << dataset_name << std::endl;
                continue;
            }
            
            std::cout << "📁 Scanning " << dataset_name << "..." << std::endl;
            int dataset_files = 0;
            
            for (const auto& entry : fs::recursive_directory_iterator(dataset_path)) {
                if (entry.is_regular_file()) {
                    std::string ext = entry.path().extension().string();
                    if (ext == ".jsonl" || ext == ".json") {
                        // Skip very small files
                        if (fs::file_size(entry.path()) < 1024) continue;
                        
                        rml_files.emplace_back(entry.path(), dataset_name);
                        dataset_files++;
                    }
                }
            }
            
            std::cout << "  Found " << dataset_files << " files in " << dataset_name << std::endl;
        }
        
        std::cout << "📁 Total: " << rml_files.size() << " RML files found" << std::endl;
        return rml_files;
    }
    
    std::string formatForTraining(const std::unordered_map<std::string, std::string>& data, 
                                 const std::string& dataset_name) {
        std::stringstream ss;
        ss << "<DATASET>" << dataset_name << "</DATASET>\n";
        
        // Add components in order
        for (const auto& component_type : rml_components) {
            auto it = data.find(component_type);
            if (it != data.end() && !it->second.empty()) {
                ss << "<" << component_type << ">" << it->second << "</" << component_type << ">\n";
            }
        }
        
        // Add original text if available
        auto text_it = data.find("text");
        if (text_it != data.end()) {
            ss << "<TEXT>" << text_it->second << "</TEXT>\n";
        }
        
        return ss.str();
    }
    
    void processFile(const fs::path& file_path, const std::string& dataset_name) {
        std::ifstream file(file_path);
        if (!file.is_open()) return;
        
        FastJsonParser parser;
        std::string line;
        int64_t file_samples = 0;
        int64_t file_duplicates = 0;
        
        // Get file size for stats
        struct stat st;
        if (stat(file_path.c_str(), &st) == 0) {
            total_size_mb += st.st_size / (1024 * 1024);
        }
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            parser.setBuffer(line);
            std::unordered_map<std::string, std::string> data;
            
            if (!parser.parseObject(data)) continue;
            
            // Extract RML components
            std::unordered_map<std::string, std::string> components;
            for (const auto& component_type : rml_components) {
                auto it = data.find(component_type);
                if (it != data.end()) {
                    components[component_type] = it->second;
                }
            }
            
            if (components.empty()) continue;
            
            // Format for training
            std::string training_text = formatForTraining(components, dataset_name);
            if (training_text.empty()) continue;
            
            // Check for duplicates
            std::string hash = hash_processor.computeHash(training_text);
            if (hash_processor.isDuplicate(hash)) {
                file_duplicates++;
                continue;
            }
            
            // Create sample
            std::stringstream sample;
            sample << "{\"text\":\"" << training_text << "\",\"dataset_source\":\"" << dataset_name << "\"}\n";
            
            // Write to appropriate split (streaming approach)
            double rand_val = dis(gen);
            std::lock_guard<std::mutex> lock(file_mutex);
            
            if (rand_val < 0.8) {
                train_file << sample.str();
            } else if (rand_val < 0.9) {
                val_file << sample.str();
            } else {
                test_file << sample.str();
            }
            
            file_samples++;
        }
        
        // Update statistics
        {
            std::lock_guard<std::mutex> lock(stats_mutex);
            total_samples += file_samples;
            duplicates_removed += file_duplicates;
            files_processed++;
            dataset_stats[dataset_name] += file_samples;
        }
        
        // Progress update every 100 files
        if (files_processed % 100 == 0) {
            std::cout << "📊 Progress: " << files_processed << " files, " << total_samples << " samples, " 
                     << duplicates_removed << " duplicates removed" << std::endl;
        }
    }
    
    void processFilesMultiThread(const std::vector<std::pair<fs::path, std::string>>& rml_files) {
        std::cout << "🚀 Processing " << rml_files.size() << " files with " << max_workers << " threads..." << std::endl;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        std::vector<std::future<void>> futures;
        
        for (const auto& file_info : rml_files) {
            futures.push_back(std::async(std::launch::async, [this, file_info]() {
                this->processFile(file_info.first, file_info.second);
            }));
            
            // Limit concurrent threads
            if (futures.size() >= max_workers) {
                for (auto& future : futures) {
                    future.wait();
                }
                futures.clear();
            }
        }
        
        // Wait for remaining threads
        for (auto& future : futures) {
            future.wait();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "✅ Processing completed in " << duration.count() / 60.0 << " minutes" << std::endl;
    }
    
    void createMetadata() {
        std::ofstream metadata_file(output_dir + "/metadata.json");
        metadata_file << "{\n";
        metadata_file << "  \"dataset_info\": {\n";
        metadata_file << "    \"name\": \"Ultra-Fast RML Training Dataset\",\n";
        metadata_file << "    \"version\": \"1.0\",\n";
        metadata_file << "    \"description\": \"Ultra-fast consolidation of 372GB RML data\"\n";
        metadata_file << "  },\n";
        metadata_file << "  \"statistics\": {\n";
        metadata_file << "    \"total_samples\": " << total_samples << ",\n";
        metadata_file << "    \"total_size_mb\": " << total_size_mb << ",\n";
        metadata_file << "    \"duplicates_removed\": " << duplicates_removed << ",\n";
        metadata_file << "    \"files_processed\": " << files_processed << "\n";
        metadata_file << "  },\n";
        metadata_file << "  \"optimization\": {\n";
        metadata_file << "    \"max_workers\": " << max_workers << ",\n";
        metadata_file << "    \"processing\": \"ultra-fast-c++\",\n";
        metadata_file << "    \"streaming\": true\n";
        metadata_file << "  }\n";
        metadata_file << "}\n";
        metadata_file.close();
    }
    
    void consolidate() {
        std::cout << "🚀 Starting ULTRA-FAST RML data consolidation..." << std::endl;
        
        auto rml_files = findRmlFiles();
        if (rml_files.empty()) {
            std::cout << "❌ No RML files found!" << std::endl;
            return;
        }
        
        processFilesMultiThread(rml_files);
        createMetadata();
        
        std::cout << "\n🎉 ULTRA-FAST RML DATA CONSOLIDATION COMPLETE!" << std::endl;
        std::cout << "📊 Total samples: " << total_samples << std::endl;
        std::cout << "📊 Duplicates removed: " << duplicates_removed << std::endl;
        std::cout << "📊 Files processed: " << files_processed << std::endl;
        std::cout << "📁 Output: " << output_dir << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "Usage: " << argv[0] << " <data_dir> <output_dir> [max_workers]" << std::endl;
        return 1;
    }
    
    std::string data_dir = argv[1];
    std::string output_dir = argv[2];
    int max_workers = (argc > 3) ? std::stoi(argv[3]) : 16;
    
    UltraFastConsolidator consolidator(data_dir, output_dir, max_workers);
    consolidator.consolidate();
    
    return 0;
} 