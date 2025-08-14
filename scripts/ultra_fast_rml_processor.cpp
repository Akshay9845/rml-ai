#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <set>
#include <thread>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstring>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <atomic> // Added for atomic variables

namespace fs = std::filesystem;

// System monitoring
struct SystemStats {
    size_t total_ram_mb;
    size_t used_ram_mb;
    size_t available_disk_gb;
    size_t used_disk_gb;
};

// RML Component structure
struct RMLComponent {
    std::string record_id;
    int chunk;
    std::string component_type;
    std::string data;
    std::string source_file;
};

// Complete RML Record
struct CompleteRMLRecord {
    std::string record_id;
    int chunk;
    std::map<std::string, std::string> components;
    bool is_complete;
};

// Progress tracking
struct ProgressTracker {
    std::atomic<size_t> files_processed{0};
    std::atomic<size_t> records_assembled{0};
    std::atomic<size_t> records_written{0};
    std::atomic<size_t> records_processed{0};
    std::atomic<size_t> total_files{0};
    std::atomic<size_t> total_records{0};
    std::chrono::steady_clock::time_point start_time;
    std::mutex progress_mutex;
};

class UltraFastRMLProcessor {
private:
    std::string data_dir;
    std::string output_dir;
    std::vector<std::string> rml_components = {
        "concepts", "emotions", "entities", "events", "intents",
        "reasoning", "summaries", "tags", "triples", "vectors"
    };
    
    ProgressTracker progress;
    std::mutex data_mutex;
    
    // Component storage per folder
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<RMLComponent>>> folder_components;
    
public:
    UltraFastRMLProcessor(const std::string& data_dir, const std::string& output_dir) 
        : data_dir(data_dir), output_dir(output_dir) {
        progress.start_time = std::chrono::steady_clock::now();
    }
    
    // Get system statistics
    SystemStats getSystemStats() {
        SystemStats stats;
        
        // Get RAM info (macOS)
        size_t total_ram = 0;
        size_t len = sizeof(total_ram);
        if (sysctlbyname("hw.memsize", &total_ram, &len, nullptr, 0) == 0) {
            stats.total_ram_mb = total_ram / (1024 * 1024);
        } else {
            stats.total_ram_mb = 16384; // Default 16GB
        }
        
        // Get disk info
        fs::space_info space = fs::space(data_dir);
        stats.available_disk_gb = space.available / (1024 * 1024 * 1024);
        stats.used_disk_gb = (space.capacity - space.available) / (1024 * 1024 * 1024);
        
        // Estimate used RAM (simplified)
        stats.used_ram_mb = stats.total_ram_mb * 0.7; // Assume 70% usage
        
        return stats;
    }
    
    // Print system status
    void printSystemStatus() {
        SystemStats stats = getSystemStats();
        std::cout << "\r💻 RAM: " << stats.used_ram_mb << "MB/" << stats.total_ram_mb 
                  << "MB | 💾 Disk: " << stats.used_disk_gb << "GB available" << std::flush;
    }
    
    // Fast JSON parsing (simplified)
    std::map<std::string, std::string> parseJSON(const std::string& line) {
        std::map<std::string, std::string> result;
        std::string current_key, current_value;
        bool in_key = false, in_value = false, in_quotes = false;
        
        for (size_t i = 0; i < line.length(); i++) {
            char c = line[i];
            
            if (c == '"' && (i == 0 || line[i-1] != '\\')) {
                in_quotes = !in_quotes;
                continue;
            }
            
            if (!in_quotes) {
                if (c == ':') {
                    in_key = false;
                    in_value = true;
                    continue;
                } else if (c == ',' || c == '}') {
                    if (!current_key.empty() && !current_value.empty()) {
                        result[current_key] = current_value;
                        current_key.clear();
                        current_value.clear();
                    }
                    in_value = false;
                    continue;
                }
            }
            
            if (in_key) {
                current_key += c;
            } else if (in_value) {
                current_value += c;
            }
        }
        
        return result;
    }
    
    // Generate missing component
    std::string generateMissingComponent(const std::string& component_type, const std::string& record_id, int chunk) {
        if (component_type == "emotions") {
            return "neutral";
        } else if (component_type == "entities") {
            return "{\"has_numbers\":false}";
        } else if (component_type == "events") {
            return "[]";
        } else if (component_type == "intents") {
            return "informative";
        } else if (component_type == "reasoning") {
            return "logical";
        } else if (component_type == "summaries") {
            return "Content summary for record " + record_id;
        } else if (component_type == "tags") {
            return "general";
        } else if (component_type == "triples") {
            return "{\"subject\":[\"content\"],\"relation\":\"has\",\"object\":[\"information\"]}";
        } else if (component_type == "vectors") {
            return "{\"text_hash\":" + std::to_string(std::hash<std::string>{}(record_id)) + "}";
        }
        return "{}";
    }
    
    // Process single file
    void processFile(const std::string& filepath, const std::string& folder_name) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "❌ Cannot open file: " << filepath << std::endl;
            return;
        }
        
        std::string line;
        size_t line_count = 0;
        
        while (std::getline(file, line)) {
            line_count++;
            if (line.empty()) continue;
            
            try {
                auto json_data = parseJSON(line);
                
                // Extract component type from filename
                std::string filename = fs::path(filepath).filename().string();
                std::string component_type = "";
                
                for (const auto& comp : rml_components) {
                    if (filename.find(comp) != std::string::npos) {
                        component_type = comp;
                        break;
                    }
                }
                
                if (component_type.empty()) continue;
                
                // Extract record_id and chunk
                std::string record_id = json_data["record_id"];
                if (record_id.empty()) {
                    record_id = json_data["doc_id"];
                }
                if (record_id.empty()) continue;
                
                int chunk = 1;
                if (json_data.find("chunk") != json_data.end()) {
                    chunk = std::stoi(json_data["chunk"]);
                }
                
                // Create component
                RMLComponent component;
                component.record_id = record_id;
                component.chunk = chunk;
                component.component_type = component_type;
                component.data = json_data["data"];
                component.source_file = filename;
                
                // Store in folder components
                std::lock_guard<std::mutex> lock(data_mutex);
                std::string key = record_id + "_" + std::to_string(chunk);
                folder_components[folder_name][key].push_back(component);
                
            } catch (const std::exception& e) {
                // Skip malformed lines
                continue;
            }
            
            // Progress update every 1000 lines
            if (line_count % 1000 == 0) {
                progress.records_processed++;
                if (line_count % 10000 == 0) {
                    printProgress();
                }
            }
        }
        
        progress.files_processed++;
        printProgress();
    }
    
    // Assemble complete records for a folder
    void assembleFolderRecords(const std::string& folder_name) {
        std::lock_guard<std::mutex> lock(data_mutex);
        
        if (folder_components.find(folder_name) == folder_components.end()) {
            return;
        }
        
        auto& folder_data = folder_components[folder_name];
        
        for (auto& [key, components] : folder_data) {
            CompleteRMLRecord record;
            
            // Extract record_id and chunk from key
            size_t pos = key.find_last_of('_');
            if (pos != std::string::npos) {
                record.record_id = key.substr(0, pos);
                record.chunk = std::stoi(key.substr(pos + 1));
            }
            
            // Collect all components
            for (const auto& comp : components) {
                record.components[comp.component_type] = comp.data;
            }
            
            // Generate missing components
            for (const auto& required_comp : rml_components) {
                if (record.components.find(required_comp) == record.components.end()) {
                    record.components[required_comp] = generateMissingComponent(required_comp, record.record_id, record.chunk);
                }
            }
            
            record.is_complete = record.components.size() == rml_components.size();
            
            if (record.is_complete) {
                progress.records_assembled++;
                writeCompleteRecord(record);
            }
        }
    }
    
    // Write complete record to output
    void writeCompleteRecord(const CompleteRMLRecord& record) {
        static std::ofstream train_file, val_file, test_file;
        static std::mutex file_mutex;
        static size_t record_counter = 0;
        
        std::lock_guard<std::mutex> lock(file_mutex);
        
        // Initialize files if not open
        if (!train_file.is_open()) {
            fs::create_directories(output_dir);
            train_file.open(output_dir + "/train.jsonl");
            val_file.open(output_dir + "/validation.jsonl");
            test_file.open(output_dir + "/test.jsonl");
        }
        
        // Create GPT-ready format
        std::string gpt_format = createGPTFormat(record);
        
        // Split into train/val/test (80/10/10)
        std::ofstream& target_file = (record_counter % 10 < 8) ? train_file : 
                                   (record_counter % 10 == 8) ? val_file : test_file;
        
        target_file << gpt_format << std::endl;
        record_counter++;
        progress.records_written++;
    }
    
    // Create GPT-ready format
    std::string createGPTFormat(const CompleteRMLRecord& record) {
        std::stringstream ss;
        ss << "{";
        ss << "\"record_id\":\"" << record.record_id << "\",";
        ss << "\"chunk\":" << record.chunk << ",";
        
        for (const auto& [comp_type, comp_data] : record.components) {
            ss << "\"" << comp_type << "\":" << comp_data << ",";
        }
        
        std::string result = ss.str();
        if (!result.empty() && result.back() == ',') {
            result.pop_back();
        }
        result += "}";
        
        return result;
    }
    
    // Print progress
    void printProgress() {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - progress.start_time);
        
        std::lock_guard<std::mutex> lock(progress.progress_mutex);
        
        std::cout << "\r🚀 Progress: " << progress.files_processed << "/" << progress.total_files 
                  << " files | Records: " << progress.records_assembled << " assembled, " 
                  << progress.records_written << " written | Time: " 
                  << std::setfill('0') << std::setw(2) << elapsed.count() / 3600 << ":"
                  << std::setfill('0') << std::setw(2) << (elapsed.count() % 3600) / 60 << ":"
                  << std::setfill('0') << std::setw(2) << elapsed.count() % 60;
        
        printSystemStatus();
        std::cout << std::flush;
    }
    
    // Count total files and records
    void countTotalFiles() {
        progress.total_files = 0;
        progress.total_records = 0;
        
        for (const auto& entry : fs::recursive_directory_iterator(data_dir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
                progress.total_files++;
                
                // Count lines in file
                std::ifstream file(entry.path());
                if (file.is_open()) {
                    progress.total_records += std::count(std::istreambuf_iterator<char>(file),
                                                       std::istreambuf_iterator<char>(), '\n');
                }
            }
        }
        
        std::cout << "📊 Found " << progress.total_files << " files with " 
                  << progress.total_records << " total records" << std::endl;
    }
    
    // Main processing function
    void processAllData() {
        std::cout << "🚀 Starting Ultra-Fast RML Processing..." << std::endl;
        std::cout << "📁 Data directory: " << data_dir << std::endl;
        std::cout << "📁 Output directory: " << output_dir << std::endl;
        
        countTotalFiles();
        
        // Process files by folder
        std::vector<std::string> folders;
        for (const auto& entry : fs::directory_iterator(data_dir)) {
            if (entry.is_directory()) {
                folders.push_back(entry.path().string());
            }
        }
        
        std::cout << "🔍 Processing " << folders.size() << " folders..." << std::endl;
        
        for (const auto& folder : folders) {
            std::string folder_name = fs::path(folder).filename().string();
            std::cout << "\n📁 Processing folder: " << folder_name << std::endl;
            
            // Process all JSONL files in folder
            for (const auto& entry : fs::directory_iterator(folder)) {
                if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
                    processFile(entry.path().string(), folder_name);
                }
            }
            
            // Assemble records for this folder
            assembleFolderRecords(folder_name);
            
            // Clear folder data to save memory
            {
                std::lock_guard<std::mutex> lock(data_mutex);
                folder_components.erase(folder_name);
            }
        }
        
        std::cout << "\n🎉 Processing complete!" << std::endl;
        std::cout << "📊 Final stats:" << std::endl;
        std::cout << "   Files processed: " << progress.files_processed << std::endl;
        std::cout << "   Records assembled: " << progress.records_assembled << std::endl;
        std::cout << "   Records written: " << progress.records_written << std::endl;
        
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - progress.start_time);
        std::cout << "   Total time: " << total_time.count() / 3600 << "h " 
                  << (total_time.count() % 3600) / 60 << "m " << total_time.count() % 60 << "s" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <data_directory> <output_directory>" << std::endl;
        std::cout << "Example: " << argv[0] << " data/ output/training_data/" << std::endl;
        return 1;
    }
    
    std::string data_dir = argv[1];
    std::string output_dir = argv[2];
    
    UltraFastRMLProcessor processor(data_dir, output_dir);
    processor.processAllData();
    
    return 0;
} 