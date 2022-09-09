#include <iostream>
#include <string>
#include <vector>

#include<sys/wait.h>

#include "json.hh"

#include "simulation.hh"

/**
 * Return a proper string representing the double passed in parameter
 * 
 * @param d double to transform in string
 * 
 * @return a string representing the double
 */
std::string doubleToString(double d) {
  std::string str = std::to_string(d);
  str.erase(str.find_last_not_of('0') + 1, std::string::npos);

  return str.back() == '.' ? str + '0' : str;
}

/**
 * Extract data from a simulation file in .tsv format and return it
 * 
 * @param path the path of the simulation file
 * 
 * @return the data contained in the simulation file
 */
std::vector<std::vector<std::string>> extractTSVData(std::string path) {
  unsigned int nCols = 7;
  std::vector<std::vector<std::string>> data(nCols);
  std::ifstream myfile;
  myfile.open(path);

  std::string line, word;
  // Don't care about the first line
  getline(myfile, line);
  while (getline(myfile, line)) {
    // Retrieve the line and store it, splitting it first by \t
    std::stringstream s(line);
    for (unsigned int i = 0; i < nCols; i++) {
      getline(s, word, '\t');
      data[i].push_back(word);
    }
  }
  myfile.close();

  return data;
}

/**
 * Read topology from a JSON file.
 * 
 * @param path string the path of the JSON file
 * 
 * @return the JSON representation of the topology
 */
Json::Value readTopology(std::string path) {
	std::ifstream jsonFile(path);
  std::string jsonText((std::istreambuf_iterator<char>(jsonFile)), std::istreambuf_iterator<char>());

  Json::CharReaderBuilder readerBuilder;
  Json::CharReader* reader = readerBuilder.newCharReader();
  Json::Value topo;
  std::string errors;
  reader->parse(jsonText.c_str(), jsonText.c_str() + jsonText.size(), &topo, &errors);

  return topo;
}

/**
 * Write some data into a file given a delimiter
 * 
 * @param data Container the data to write
 * @param file std::ofstream& the file to write in
 * @param delimiter string the delimiter between each piece of data
 */
void writeInFile(const std::vector<std::string>& data, std::ofstream& file, const std::string& delimiter) {
  std::stringstream s;
  copy(data.begin(), data.end(), std::ostream_iterator<std::string>(s, delimiter.c_str()));
  std::string content = s.str();
  content = content.substr(0, content.size() - 1);
  file << content << std::endl;
}

/**
 * Gather all the simulation result into a single file
 * 
 * @param path string the path of the file to write the aggregated result
 * @param temp string the template of the simulation files
 * @param nSim unsigned int the number of simulation files
 * @param nTests unsigned int the number of tests in a single simulation
 */
void aggregateSimulationsResults(std::string path, std::string temp, unsigned int nSim, unsigned int nTests) {
  std::ofstream rewFile, fairFile, cumFile, apsFile, stasFile, confFile, persFile;
  rewFile.open(path+"_rew.tsv");
  fairFile.open(path+"_fair.tsv");
  cumFile.open(path+"_cum.tsv");
  apsFile.open(path+"_aps.tsv");
  stasFile.open(path+"_stas.tsv");
  persFile.open(path+"_pers.tsv");
  confFile.open(path+"_conf.tsv");

  // First line of aggregation
  std::string delimiter = "\t";
  std::string firstLine = "";
  for (unsigned int i = 0; i < nTests; i++) firstLine += std::to_string(i) + (i < nTests - 1 ? delimiter : "");
  rewFile << firstLine << std::endl;
  fairFile << firstLine << std::endl;
  cumFile << firstLine << std::endl;
  apsFile << firstLine << std::endl;
  stasFile << firstLine << std::endl;
  persFile << firstLine << std::endl;
  confFile << firstLine << std::endl;

  // Each line is a simulation
  for (unsigned int i = 0; i < nSim; i++) {
    std::vector<std::vector<std::string>> data = extractTSVData(temp + std::to_string(i) + ".tsv");
    writeInFile(data[0], rewFile, delimiter); // Reward
    writeInFile(data[1], fairFile, delimiter); // Fairness
    writeInFile(data[2], cumFile, delimiter); // Cum
    writeInFile(data[3], apsFile, delimiter); // APs
    writeInFile(data[4], stasFile, delimiter); // STAs
    writeInFile(data[5], persFile, delimiter); // PERs
    writeInFile(data[6], confFile, delimiter); // Configurations   

    // Remove the data file
    remove((temp + std::to_string(i) + ".tsv").c_str());
  }

  rewFile.close();
  fairFile.close();
  cumFile.close();
  apsFile.close();
  stasFile.close();
  confFile.close();
}

int main (int argc, char *argv[]) {
  std::vector<bool> uplinks = {true};
  // Number of simulations to run
  unsigned int nSimulations = 22;
  // Duration of a single simulation
  double duration = 120;
  // Duration of a single test
  std::vector<double> testDurations({0.075});
  // std::vector<double> dists({5});
  // Optimizers to test
  std::vector<Optim> optimizers({EGREEDY, THOMP_NORM, IDLEOPT, MARGIN, THOMP_GAMNORM}); // EGREEDY, THOMP_NORM, IDLEOPT, MARGIN
  // Samplers to test
  std::vector<Samp> samplers({UNIF, HGM, HCM}); // HCM, , UNIF, HGM
  // Entries
  std::vector<Entry> entries({DEF, DEGA}); // DEGA, DEF
  // Rewards to test
  std::vector<Reward> rewards({AD_HOC});
  // Distance mode to test
  std::vector<DistanceMode> dmodes({STD});
  // Channel width
  std::vector<ChannelWidth> channels({MHZ_20});
  // Default configurations
  std::vector<NetworkConfiguration> defaultConfs;
  std::vector<std::vector<double>> vec_defaultConfs = {}; // {-78,10, -74,4, -77,9, -79,8, -78,11, -73,6, -76,9, -79,5, -76,5, -77,4}
  for (unsigned int i = 0; i < vec_defaultConfs.size(); i++) {
    NetworkConfiguration nc;
    for (unsigned int j = 0; j < vec_defaultConfs[i].size(); j += 2) {
      nc.push_back(std::tuple<double, double>(vec_defaultConfs[i][j], vec_defaultConfs[i][j+1]));
    }
    defaultConfs.push_back(nc);
  }
  // Station throughput
  std::vector<std::vector<unsigned int>> predef_throughputs = {};
  std::vector<double> stations_types({0.0, 0, 0, 1.0});

  std::discrete_distribution<int> discreteDist(stations_types.begin(), stations_types.end());
  std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
  // Topos to test
  int topoIndex = -1;
  std::vector<std::string> topos({"C6o", "MER_FLOORS_CH20_S5"}); // MER_FLOORS_BAD_DIM
  for (std::string topo: topos) {
    // Default configuration
    topoIndex++;
    NetworkConfiguration def;
    if (topoIndex < (int) defaultConfs.size()) {
      def = defaultConfs[topoIndex];
    }
    // Reading the topology object to get the number of stations
    Json::Value topology = readTopology("./scratch/nsTest/topos/" + topo + ".json");
    
    // Attribute stations types
    std::vector<StationThroughput> stations_throughputs;
    if (topoIndex < (int) predef_throughputs.size()) {
      for (unsigned int ui : predef_throughputs[topoIndex]) {
        stations_throughputs.push_back(static_cast<StationThroughput>(ui));
      }
    } else {
      for (Json::Value sta : topology["stations"]) {
        int st = discreteDist(generator);
        stations_throughputs.push_back(static_cast<StationThroughput>(st));
      }
    }
    std::string stId = "";
    for (StationThroughput sta : stations_throughputs) {
      stId += std::to_string(sta);
    }
    for (bool uplink: uplinks) {
      for (DistanceMode dm: dmodes) {
        for (Entry entry: entries) {
          for (double testDuration: testDurations) {
            // For each channel
            for (ChannelWidth cw: channels) {
              // For each optimizer
              for (Optim o: optimizers) {
                // For each sampler
                for (Samp s: samplers) {
                  // For each reward
                  for (Reward r: rewards) {
                    // Find the right identifiers
                    // Don't test some combinations
                    if ((topo == "MER_FLOORS_CH40_S5" && cw != MHZ_40) || ((topo == "MER_FLOORS_CH20_S5" || topo == "MER_FLOORS_BAD_DIM") && cw != MHZ_20) || ((o == EGREEDY || o == MARGIN || o == IDLEOPT || o == THOMP_NORM) && (s != UNIF || entry != DEF)) || (o == THOMP_GAMNORM && (s == UNIF || (s == HGM && entry != DEF) || (s == HCM && entry == DEF)))) { // || (topo == "T7" && s == HGM) || (topo == "T12" && o == EGREEDY)) {
                      continue;
                    }
                    std::string oId = "", sId = "", rId = "", eId = "", dId = "", cId = "";
                    switch (o) {
                      case IDLEOPT: oId = "IDLE"; break;
                      case THOMP_GAMNORM: oId = "TGNORM"; break;
                      case THOMP_NORM: oId = "TNORM"; break;
                      case EGREEDY: oId = "EGREED"; break;
                      case MARGIN: oId = "MARGIN"; break;
                      case RANDNEIGHBOR: oId = "RANDNEIGH"; break;
                    }

                    switch (s) {
                      case HGM: sId = "HGMT"; break;
                      case UNIF: sId = "UNI"; break;
                      case HCM: sId = "HCM"; break;
                    }

                    switch (r) {
                      case AD_HOC: rId = "ADHOC"; break;
                      case CUMTP: rId = "CUMTP"; break;
                      case LOGPF: rId = "LOGPF"; break;
                    }

                    switch (entry) {
                      case DEF: eId = "DEF"; break;
                      case DEGA: eId = "DEGA"; break;
                    }

                    switch (dm) {
                      case STD: dId = "STD"; break;
                      case CYCLE: dId = "CYCLE"; break;
                    }

                    switch (cw) {
                      case MHZ_20: cId = "20"; break;
                      case MHZ_40: cId = "40"; break;
                      case MHZ_80: cId = "80"; break;
                    }

                    // Build the template of the output
                    std::string traffic_type = uplink ? "BOTH" : "DOWN";
                    std::string outputName = topo + "_" + stId + "_" + cId + "_" + eId + "_" + dId + "_" + doubleToString(duration) + "_" + oId + "_" + sId + "_" + rId + "_" + doubleToString(testDuration) + "_" + traffic_type + "_MINSTREL";
                    // std::string outputName = topo + "_" + stId + deg + cId + "_" + eId + "_" + dId + "_" + doubleToString(duration) + "_0.9_" + doubleToString(testDuration);

                    // Log
                    std::cout << "Working on " << outputName << "..." << std::endl;

                    // Launching nSimulations simulations
                    std::vector<Simulation*> simulations(nSimulations);
                    for (unsigned int i = 0; i < nSimulations; i++) {
                      usleep(1000);
                      simulations[i] = new Simulation(o, s, r, entry, dm, cw, topology, stations_throughputs, duration, testDuration, uplink, outputName + "_" + std::to_string(i) + ".tsv", def);
                    }

                    // Wait for all simulations to finish
                    for (unsigned int i = 0; i < nSimulations; i++) {
                      waitpid(simulations[i]->getPID(), NULL, 0);
                      delete simulations[i];
                    }
                    std::cout << "Simulations terminated" << std::endl;

                    // Aggregate the data
                    aggregateSimulationsResults("./scratch/nsTest/data/"+outputName, "./scratch/nsTest/data/"+outputName+"_",
                                                nSimulations, duration / testDuration);
                    std::cout << "Aggregation terminated" << std::endl << std::endl;
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}