#ifndef __SIMULATION_HH
#define __SIMULATION_HH

#include <string>
#include <vector>
#include <set>
#include "unistd.h"
#include <sys/time.h>

#include "json.hh"

#include "ns3/core-module.h"
#include "ns3/network-module.h"
#include "ns3/mobility-module.h"
#include "ns3/config-store-module.h"
#include "ns3/wifi-module.h"
#include "ns3/internet-module.h"
#include "ns3/applications-module.h"

#include "optimizers.hh"
#include "samplers.hh"

using namespace ns3;

enum Optim { IDLEOPT, EGREEDY, THOMP_GAMNORM, THOMP_NORM, MARGIN, RANDNEIGHBOR };
enum Samp { UNIF, HGM, HCM };
enum Reward { AD_HOC, CUMTP, LOGPF };
enum Entry { DEF, DEGA };
enum Dist { LOG, SQRT, N2, N4 };
enum ChannelWidth { MHZ_20, MHZ_40, MHZ_80 };
enum StationThroughput { NONE=0, LOW=1, MEDIUM=2, HIGH=3 };

class Simulation {
	public:
		Simulation(Optim oId, Samp sId, Reward r, Entry e, DistanceMode dmode, ChannelWidth cw, Json::Value topo, std::vector<StationThroughput> stations_throughputs, double duration, double testDuration, bool uplink, std::string outputName, NetworkConfiguration defaultConf = NetworkConfiguration());
		pid_t getPID() const;
		void readTopology(Json::Value topo);
		void storeMetrics();
		double rewardFromThroughputs();
		double rewardFromThroughputs(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables);
		double adHocReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables);
		double logPfReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables);
		double cumulatedThroughputReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables);
		double cumulatedThroughputFromThroughputs();
		std::vector<std::tuple<double, unsigned int>> subrewardFromThroughputs();
		double fairnessFromThroughputs();
		std::vector<NetworkConfiguration> findDiagonalEntryPoints(unsigned int n) const;
		std::vector<NetworkConfiguration> findDegreeEntryPoints(double criterion=0.5) const;
		std::vector<NetworkConfiguration> findNHDegreeEntryPoints(double criterion=0.5) const;
		NetworkConfiguration handleClusterizedConfiguration(const NetworkConfiguration& configuration);
		std::vector<double> getLogDistanceRSSIs() const;
		std::vector<double> apThroughputsFromThroughputs();
		std::vector<double> staThroughputsFromThroughputs();
		std::vector<double> staPersFromPers();
		std::vector<std::vector<double>> attainableThroughputs();
		void computeThroughputsAndErrors();
		std::vector<NetworkConfiguration> findEntryPoints(int v = -1) const;
		std::vector<std::vector<unsigned int>> extractConflicts(NetworkConfiguration conf) const;
		void setupNewConfiguration(NetworkConfiguration configuration);
		void endOfTest();
		void stationsThroughputsToInterval(const std::vector<StationThroughput>& stations_throughputs, double duration);
		static std::string configurationToString(const NetworkConfiguration& config);
		static int channelNumber(ChannelWidth cw);
		static bool parameterConstraint(double sens, double pow);
		static unsigned int numberOfSamples(std::vector<GaussianT> gaussians);
		static unsigned int numberOfSamples_HCM(std::vector<Ring> circulars);
		static WifiNetDevice* getWifiDevice(Node* node);
		static WifiMac* getMAC(Node* node);
		static WifiMacHeader createAdHocMacHeader(Node* from, Node* to);
		static WifiMode getWifiMode(Node* from, Node* to);
		static unsigned int getMCSValue(Node* from, Node* to);
		static std::string getMCSClass(Node* from, Node* to);
		static double pathLoss(std::tuple<double, double, double> source, std::tuple<double, double, double> target, double txPower);
		static double distance(std::tuple<double, double, double> source, std::tuple<double, double, double> target);
		static std::vector<double> attainableThroughputsFromChannel(ChannelWidth cw);

	protected:
		pid_t _pid;
		Reward _rewardType;
		bool _changed;
		double _testDuration;
		unsigned int _testCounter;
		unsigned int _warmup_tests;
		std::vector<double> _rewards;
		std::vector<double> _fairness;
		std::vector<double> _cumulatedThroughput;
		std::vector<NetworkConfiguration> _configurations;
		std::vector<NetworkConfiguration> _entryPoints;
		std::vector<std::vector<double>> _apThroughputs;
		std::vector<std::vector<double>> _staThroughputs;
		std::vector<std::vector<double>> _staPERs;
		std::vector<double> _intervals;
		std::vector<double> _positionAPX;
		std::vector<double> _positionAPY;
		std::vector<double> _positionAPZ;
		std::vector<double> _positionStaX;
		std::vector<double> _positionStaY;
		std::vector<double> _positionStaZ;
		std::vector<unsigned int> _clustersAP;
		std::vector<std::vector<double>> _throughputs;
		std::vector<std::vector<double>> _pers;
		std::vector<std::vector<unsigned int>> _associations;
		std::vector<NetDeviceContainer> _devices;
		NetworkConfiguration _configuration;
		Optimizer* _optimizer;
		NodeContainer _nodesAP;
		std::vector<NodeContainer> _nodesSta;
		std::vector<ApplicationContainer> _serversPerAp;
		std::vector<std::vector<unsigned int>> _lastRxPackets;
		std::vector<std::vector<unsigned int>> _lastLostPackets;
		ChannelWidth _channel_width;
		std::vector<double> _heAttainableThroughputs = std::vector<double>({
			120e6, 230e6, 330e6, 430e6, 610e6, 780e6, 850e6, 890e6, 1070e6, 1180e6, 1220e6, 1400e6
		});
		unsigned int _packetSize = 8 * 1464;
		int _defaultSensibility = -82;
		int _defaultPower = 20;
		bool _warmed = false;
		double _cumulative = 0.0;
		double _ema = -1.0;
};

#endif