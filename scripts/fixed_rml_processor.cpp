#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <chrono>
#include <filesystem>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <thread>
#include <atomic>
#include <sys/stat.h>
#include <sys/sysctl.h>

namespace fs = std::filesystem;

// Progress tracking per folder
struct FolderProgress {
    std::string folder_name;
    size_t total_files;
    size_t processed_files;
    size_t total_records;
    size_t processed_records;
    size_t assembled_records;
    std::chrono::steady_clock::time_point start_time;
    double speed_mb_per_sec;
    double eta_seconds;
};

// RML Component
struct RMLComponent {
    std::string record_id;
    int chunk;
    std::string component_type;
    std::string data;
};

// Complete RML Record
struct CompleteRMLRecord {
    std::string record_id;
    int chunk;
    std::map<std::string, std::string> components;
};

class FixedRMLProcessor {
private:
    std::string data_dir;
    std::string output_dir;
    std::vector<std::string> rml_components = {
        "concepts", "emotions", "entities", "events", "intents",
        "reasoning", "summaries", "tags", "triples", "vectors"
    };
    
    std::atomic<size_t> global_files_processed{0};
    std::atomic<size_t> global_records_assembled{0};
    std::atomic<size_t> global_records_written{0};
    std::chrono::steady_clock::time_point global_start_time;
    
public:
    FixedRMLProcessor(const std::string& data_dir, const std::string& output_dir) 
        : data_dir(data_dir), output_dir(output_dir) {
        global_start_time = std::chrono::steady_clock::now();
        fs::create_directories(output_dir);
    }
    
    // Fast JSON parsing
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
    std::string generateMissingComponent(const std::string& component_type, const std::string& record_id) {
        if (component_type == "emotions") return "neutral";
        if (component_type == "entities") return "{\"has_numbers\":false}";
        if (component_type == "events") return "[]";
        if (component_type == "intents") return "informative";
        if (component_type == "reasoning") return "logical";
        if (component_type == "summaries") return "Content summary for " + record_id;
        if (component_type == "tags") return "general";
        if (component_type == "triples") return "{\"subject\":[\"content\"],\"relation\":\"has\",\"object\":[\"information\"]}";
        if (component_type == "vectors") return "{\"text_hash\":" + std::to_string(std::hash<std::string>{}(record_id)) + "}";
        return "{}";
    }
    
    // Extract component type from filename
    std::string extractComponentType(const std::string& filename) {
        std::string lower_filename = filename;
        std::transform(lower_filename.begin(), lower_filename.end(), lower_filename.begin(), ::tolower);
        
        for (const auto& comp : rml_components) {
            if (lower_filename.find(comp) != std::string::npos) {
                return comp;
            }
        }
        
        // Check for batch files
        if (lower_filename.find("batch") != std::string::npos) {
            for (const auto& comp : rml_components) {
                if (lower_filename.find(comp) != std::string::npos) {
                    return comp;
                }
            }
        }
        
        // Check for chunk files
        if (lower_filename.find("chunk") != std::string::npos) {
            for (const auto& comp : rml_components) {
                if (lower_filename.find(comp) != std::string::npos) {
                    return comp;
                }
            }
        }
        
        return "";
    }
    
    // Process single file with speed tracking
    size_t processFile(const std::string& filepath, const std::string& folder_name, 
                      std::unordered_map<std::string, std::vector<RMLComponent>>& folder_data) {
        std::ifstream file(filepath);
        if (!file.is_open()) return 0;
        
        size_t records_processed = 0;
        std::string line;
        
        // Extract component type from filename
        std::string filename = fs::path(filepath).filename().string();
        std::string component_type = extractComponentType(filename);
        
        if (component_type.empty()) {
            std::cout << "   ⚠️ Could not determine component type for: " << filename << std::endl;
            return 0;
        }
        
        while (std::getline(file, line)) {
            if (line.empty()) continue;
            
            try {
                auto json_data = parseJSON(line);
                
                // Extract record_id and chunk - try multiple field names
                std::string record_id = "";
                if (json_data.find("record_id") != json_data.end()) {
                    record_id = json_data["record_id"];
                } else if (json_data.find("doc_id") != json_data.end()) {
                    record_id = json_data["doc_id"];
                } else if (json_data.find("document_id") != json_data.end()) {
                    record_id = json_data["document_id"];
                }
                
                if (record_id.empty()) continue;
                
                int chunk = 1;
                if (json_data.find("chunk") != json_data.end()) {
                    chunk = std::stoi(json_data["chunk"]);
                }
                
                // Extract data - try multiple field names
                std::string data = "";
                if (json_data.find("data") != json_data.end()) {
                    data = json_data["data"];
                } else if (json_data.find("concepts") != json_data.end()) {
                    data = json_data["concepts"];
                } else if (json_data.find("summary") != json_data.end()) {
                    data = json_data["summary"];
                } else if (json_data.find("text") != json_data.end()) {
                    data = json_data["text"];
                } else if (json_data.find("tone") != json_data.end()) {
                    data = json_data["tone"];
                } else if (json_data.find("category") != json_data.end()) {
                    data = json_data["category"];
                } else if (json_data.find("subject") != json_data.end()) {
                    data = json_data["subject"];
                }
                
                // Create component
                RMLComponent component;
                component.record_id = record_id;
                component.chunk = chunk;
                component.component_type = component_type;
                component.data = data;
                
                // Store in folder data
                std::string key = record_id + "_" + std::to_string(chunk);
                folder_data[key].push_back(component);
                records_processed++;
                
            } catch (const std::exception& e) {
                continue;
            }
        }
        
        return records_processed;
    }
    
    // Assemble complete records for a folder
    size_t assembleFolderRecords(const std::string& folder_name, 
                                const std::unordered_map<std::string, std::vector<RMLComponent>>& folder_data) {
        size_t assembled_count = 0;
        std::ofstream output_file(output_dir + "/" + folder_name + "_complete.jsonl");
        
        for (const auto& [key, components] : folder_data) {
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
                    record.components[required_comp] = generateMissingComponent(required_comp, record.record_id);
                }
            }
            
            // Write complete record
            if (record.components.size() == rml_components.size()) {
                std::string gpt_format = createGPTFormat(record);
                output_file << gpt_format << std::endl;
                assembled_count++;
            }
        }
        
        return assembled_count;
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
    
    // Print folder progress
    void printFolderProgress(const FolderProgress& progress) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - progress.start_time);
        
        double progress_percent = (progress.processed_files * 100.0) / progress.total_files;
        double speed = progress.speed_mb_per_sec;
        
        std::cout << "\r📁 " << std::setw(20) << std::left << progress.folder_name 
                  << " | " << std::setw(3) << std::right << (int)progress_percent << "%"
                  << " | Files: " << progress.processed_files << "/" << progress.total_files
                  << " | Records: " << progress.assembled_records
                  << " | Speed: " << std::fixed << std::setprecision(1) << speed << " MB/s"
                  << " | Time: " << std::setfill('0') << std::setw(2) << elapsed.count() / 60 << ":"
                  << std::setfill('0') << std::setw(2) << elapsed.count() % 60
                  << std::flush;
    }
    
    // Process single folder
    void processFolder(const std::string& folder_path) {
        std::string folder_name = fs::path(folder_path).filename().string();
        
        // Count files and records
        std::vector<std::string> jsonl_files;
        size_t total_records = 0;
        
        for (const auto& entry : fs::directory_iterator(folder_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".jsonl") {
                jsonl_files.push_back(entry.path().string());
                
                // Quick line count
                std::ifstream file(entry.path());
                if (file.is_open()) {
                    total_records += std::count(std::istreambuf_iterator<char>(file),
                                              std::istreambuf_iterator<char>(), '\n');
                }
            }
        }
        
        if (jsonl_files.empty()) {
            std::cout << "📁 " << std::setw(20) << std::left << folder_name << " | No JSONL files found" << std::endl;
            return;
        }
        
        // Initialize progress
        FolderProgress progress;
        progress.folder_name = folder_name;
        progress.total_files = jsonl_files.size();
        progress.total_records = total_records;
        progress.start_time = std::chrono::steady_clock::now();
        
        std::cout << "\n🚀 Processing folder: " << folder_name 
                  << " (" << progress.total_files << " files, " << progress.total_records << " records)" << std::endl;
        
        // Process files
        std::unordered_map<std::string, std::vector<RMLComponent>> folder_data;
        size_t total_size_mb = 0;
        
        for (const auto& filepath : jsonl_files) {
            // Get file size
            std::ifstream file(filepath, std::ios::binary | std::ios::ate);
            if (file.is_open()) {
                total_size_mb += file.tellg() / (1024 * 1024);
            }
            
            // Process file
            size_t records = processFile(filepath, folder_name, folder_data);
            progress.processed_files++;
            progress.processed_records += records;
            
            // Calculate speed
            auto now = std::chrono::steady_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - progress.start_time);
            if (elapsed.count() > 0) {
                progress.speed_mb_per_sec = total_size_mb / elapsed.count();
            }
            
            // Print progress every 5 files
            if (progress.processed_files % 5 == 0 || progress.processed_files == progress.total_files) {
                printFolderProgress(progress);
            }
            
            global_files_processed++;
        }
        
        // Assemble records
        progress.assembled_records = assembleFolderRecords(folder_name, folder_data);
        global_records_assembled += progress.assembled_records;
        
        // Final progress
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - progress.start_time);
        
        std::cout << "\n✅ " << std::setw(20) << std::left << folder_name 
                  << " | Completed in " << total_time.count() << "s"
                  << " | " << progress.assembled_records << " complete records"
                  << " | Avg speed: " << std::fixed << std::setprecision(1) 
                  << (total_size_mb / total_time.count()) << " MB/s" << std::endl;
    }
    
    // Main processing function
    void processAllFolders() {
        std::cout << "🚀 Fixed RML Processor - Processing Each Folder" << std::endl;
        std::cout << "📁 Data directory: " << data_dir << std::endl;
        std::cout << "📁 Output directory: " << output_dir << std::endl;
        std::cout << "=" << std::string(80, '=') << std::endl;
        
        // Get all folders
        std::vector<std::string> folders;
        for (const auto& entry : fs::directory_iterator(data_dir)) {
            if (entry.is_directory()) {
                folders.push_back(entry.path().string());
            }
        }
        
        std::cout << "📊 Found " << folders.size() << " folders to process" << std::endl;
        std::cout << "=" << std::string(80, '=') << std::endl;
        
        // Process each folder
        for (const auto& folder : folders) {
            processFolder(folder);
        }
        
        // Final summary
        auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::steady_clock::now() - global_start_time);
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "🎉 PROCESSING COMPLETE!" << std::endl;
        std::cout << "📊 Final Statistics:" << std::endl;
        std::cout << "   Files processed: " << global_files_processed << std::endl;
        std::cout << "   Records assembled: " << global_records_assembled << std::endl;
        std::cout << "   Total time: " << total_time.count() / 3600 << "h " 
                  << (total_time.count() % 3600) / 60 << "m " << total_time.count() % 60 << "s" << std::endl;
        std::cout << "📁 Check " << output_dir << " for complete RML records" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Usage: " << argv[0] << " <data_directory> <output_directory>" << std::endl;
        std::cout << "Example: " << argv[0] << " data/ output/fixed_processing/" << std::endl;
        return 1;
    }
    
    std::string data_dir = argv[1];
    std::string output_dir = argv[2];
    
    FixedRMLProcessor processor(data_dir, output_dir);
    processor.processAllFolders();
    
    return 0;
} 