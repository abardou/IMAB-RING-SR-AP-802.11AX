#include "simulation.hh"
#include <iostream>

/**
 * Create a simulation in a dedicated process
 * 
 * @param oId Optim the optimizer to use
 * @param sId Samp the sampler to use
 * @param r Reward the reward to use
 * @param topo Json::value the JSON representation of the network topology
 * @param duration double the simulation duration
 * @param testDuration double the test duration
 * @param outputName std::string the output file name
 * @param beta double the beta parameter for FSCORE reward
 */ 
Simulation::Simulation(Optim oId, Samp sId, Reward r, Entry e, DistanceMode dmode, ChannelWidth cw, Json::Value topo, std::vector<StationThroughput> stations_throughputs, double duration, double testDuration, bool uplink, std::string outputName, NetworkConfiguration defaultConf) : _rewardType(r), _changed(false), _testDuration(testDuration), _testCounter(0), _channel_width(cw) {
	this->_pid = fork();
	if (this->_pid == 0) {
		// Child process

		// Random generator
		struct timeval time; 
    gettimeofday(&time,NULL);
		unsigned int seed = (time.tv_sec * 10) + (time.tv_usec / 10);
		RngSeedManager::SetSeed(seed);

		// Topology
		this->readTopology(topo);

		// Default configuration
		if (this->_positionAPX.size() != defaultConf.size()) {
			defaultConf.clear();
			for (unsigned int i = 0; i < this->_positionAPX.size(); i++) {
				defaultConf.push_back(std::tuple<double, double>(this->_defaultSensibility, this->_defaultPower));
			}
		}

		// Index of channel to use during the simulation
		int channel = Simulation::channelNumber(this->_channel_width), numberOfAPs = this->_positionAPX.size(), numberOfStas = 0;
		for (std::vector<unsigned int> assocs: this->_associations)
			numberOfStas += assocs.size();

		double warmup_time = 2 * this->_testDuration;
		this->_warmup_tests = ceil(warmup_time / this->_testDuration);
		
		double applicationStart = 2.0, applicationEnd = applicationStart + duration + this->_warmup_tests * this->_testDuration;

		//Adapt interval according to specified station throughputs
		stationsThroughputsToInterval(stations_throughputs, duration);
		// for (int i = 0; i < numberOfAPs; i++) intervalsCross[i] = this->_intervalCross * this->_associations[i].size();

		// APs creation and configuration
		// At the start, they're all configured with 802.11 default conf
		this->_nodesAP.Create(numberOfAPs);
		std::vector<YansWifiPhyHelper> wifiPhy(numberOfAPs); // One PHY for each AP
		for(int i = 0; i < numberOfAPs; i++) {
			wifiPhy[i] = YansWifiPhyHelper::Default();
			wifiPhy[i].Set("Antennas", UintegerValue(2));
			// 2 spatial streams to support htMcs from 8 to 15 with short GI
			wifiPhy[i].Set("MaxSupportedTxSpatialStreams", UintegerValue(2));
			wifiPhy[i].Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
			wifiPhy[i].Set("ChannelNumber", UintegerValue(channel));
			wifiPhy[i].Set("RxSensitivity", DoubleValue(std::get<0>(defaultConf[i])));
			wifiPhy[i].Set("CcaEdThreshold", DoubleValue(std::get<0>(defaultConf[i])));
			wifiPhy[i].Set("TxPowerStart", DoubleValue(std::get<1>(defaultConf[i])));
			wifiPhy[i].Set("TxPowerEnd", DoubleValue(std::get<1>(defaultConf[i])));
		}

		// Stations creation and configuration
		this->_nodesSta = std::vector<NodeContainer>(numberOfAPs);
		for(int i = 0; i < numberOfAPs; i++) this->_nodesSta[i].Create(this->_associations[i].size());
		// One phy for every station
		YansWifiPhyHelper staWifiPhy = YansWifiPhyHelper::Default();
		staWifiPhy.Set("Antennas", UintegerValue(2));
		staWifiPhy.Set("MaxSupportedTxSpatialStreams", UintegerValue(2));
		staWifiPhy.Set("MaxSupportedRxSpatialStreams", UintegerValue(2));
		staWifiPhy.Set("ChannelNumber", UintegerValue(channel));
		staWifiPhy.Set("RxSensitivity", DoubleValue(this->_defaultSensibility));
		staWifiPhy.Set("CcaEdThreshold", DoubleValue(this->_defaultSensibility));

		// Propagation model, same for everyone
		YansWifiChannelHelper wifiChannel;
		wifiChannel.SetPropagationDelay ("ns3::ConstantSpeedPropagationDelayModel");
		wifiChannel.AddPropagationLoss ("ns3::LogDistancePropagationLossModel");
		Ptr<YansWifiChannel> channelPtr = wifiChannel.Create ();
		// Attribution to stations and models
		staWifiPhy.SetChannel(channelPtr);
		for(int i = 0; i < numberOfAPs; i++) wifiPhy[i].SetChannel(channelPtr);

		// 802.11ax protocol
		WifiHelper wifi;
		wifi.SetStandard (WIFI_PHY_STANDARD_80211ax_5GHZ);
		// TOCHANGE, Vht pour ax (0 pour le contrôle, à fixer pour les données)
		wifi.SetRemoteStationManager("ns3::IdealWifiManager"); // Minstrel ARF AARF // wifi.SetRemoteStationManager("ns3::ConstantRateWifiManager", "DataMode", StringValue ("HeMcs4"), "ControlMode", StringValue ("HeMcs0"));


		// Configure Infrastructure mode and SSID
		std::vector<NetDeviceContainer> devices(numberOfAPs);//Un groupe de Sta par AP
		Ssid ssid = Ssid ("ns380211");

		// Mac for Stations
		WifiMacHelper wifiMac;
		wifiMac.SetType("ns3::StaWifiMac", "Ssid", SsidValue(ssid));
		for (int i = 0; i < numberOfAPs; i++) devices[i] = wifi.Install(staWifiPhy, wifiMac, this->_nodesSta[i]);
		// Mac for APs
		this->_devices = std::vector<NetDeviceContainer>(numberOfAPs);
		wifiMac.SetType("ns3::ApWifiMac", "Ssid", SsidValue(ssid));
		for (int i = 0; i < numberOfAPs; i++) this->_devices[i] = wifi.Install(wifiPhy[i], wifiMac, this->_nodesAP.Get(i));

		// Mobility for devices
		MobilityHelper mobility;
		Ptr<ListPositionAllocator> positionAlloc = CreateObject<ListPositionAllocator>();
		// For APs
		for (int i = 0; i < numberOfAPs; i++) positionAlloc->Add(Vector(this->_positionAPX[i], this->_positionAPY[i], this->_positionAPZ[i]));
		// For stations
		for (int i = 0; i < numberOfAPs; i++)
			for (unsigned int j = 0 ; j < this->_associations[i].size(); j++)
				positionAlloc->Add(Vector(this->_positionStaX[this->_associations[i][j]], this->_positionStaY[this->_associations[i][j]], this->_positionStaZ[this->_associations[i][j]]));
		// Devices are not moving
		mobility.SetPositionAllocator(positionAlloc);
		mobility.SetMobilityModel("ns3::ConstantPositionMobilityModel");
		// Application to APs
		mobility.Install(this->_nodesAP);
		// Application to stations
		for(int i = 0; i < numberOfAPs; i++) mobility.Install(this->_nodesSta[i]);

		//IP stack and addresses
		InternetStackHelper internet;
		for(int i = 0; i < numberOfAPs; i++) internet.Install(this->_nodesSta[i]);
		internet.Install(this->_nodesAP);

		Ipv4AddressHelper ipv4;
		ipv4.SetBase("10.1.0.0", "255.255.0.0");
		Ipv4InterfaceContainer apInterfaces;
		for(int i = 0; i < numberOfAPs; i++) {
			apInterfaces = ipv4.Assign(this->_devices[i]);
			ipv4.Assign(devices[i]);
		}

		// Traffic configuration
		// Server is installed on all stations
		uint16_t port = 4000;
		UdpServerHelper server(port);
		this->_serversPerAp = std::vector<ApplicationContainer>(numberOfAPs);
		for(int i = 0; i < numberOfAPs; i++) {
			ApplicationContainer apps = server.Install(this->_nodesSta[i]);
			apps.Start(Seconds(0));
			apps.Stop(Seconds(applicationEnd));
			this->_serversPerAp[i] = apps;
		}

		if (uplink) {
				ApplicationContainer apps = server.Install(this->_nodesAP);
				apps.Start(Seconds(0));
				apps.Stop(Seconds(applicationEnd));
			}
		
		// Client is installed on all APs
		UdpClientHelper clientCT;//CT=cross traffic (from AP to stations)
		UdpClientHelper clientCTARP;//CT=cross traffic (from AP to stations) with small throughput to force ARP tables to fill
		unsigned sta_idx = 0;

		double step = applicationStart / (numberOfStas + 1.0);
		for(int i = 0; i < numberOfAPs; i++) {
				for(unsigned int j = 0; j < this->_associations[i].size(); j++) {
						// IPv4 instance of the station
						Ipv4Address addr = this->_nodesSta[i].Get(j)
							->GetObject<Ipv4>()
							->GetAddress(1, 0) // Loopback (1-0)
							.GetLocal();

						clientCTARP.SetAttribute ("RemoteAddress",AddressValue(addr));
						clientCTARP.SetAttribute ("RemotePort",UintegerValue(port));
						clientCTARP.SetAttribute ("MaxPackets", UintegerValue(2));
						clientCTARP.SetAttribute ("Interval", TimeValue(Seconds(step / 2.0)));
						clientCTARP.SetAttribute ("PacketSize", UintegerValue(this->_packetSize / 8.0));

						clientCT.SetAttribute ("RemoteAddress",AddressValue(addr));
						clientCT.SetAttribute ("RemotePort",UintegerValue(port));
						clientCT.SetAttribute ("MaxPackets", UintegerValue(1e9));
						clientCT.SetAttribute ("Interval", TimeValue(Seconds(this->_intervals[this->_associations[i][j]])));
						clientCT.SetAttribute ("PacketSize", UintegerValue(this->_packetSize / 8.0));

						// Installation on AP
						ApplicationContainer apps_arp = clientCTARP.Install(this->_nodesAP.Get(i));
						apps_arp.Start(Seconds(sta_idx * step));
						apps_arp.Stop(Seconds((sta_idx + 1) * step));
						sta_idx++;

						ApplicationContainer apps = clientCT.Install(this->_nodesAP.Get(i));
						apps.Start(Seconds(applicationStart));
						apps.Stop(Seconds(applicationEnd));

						// If uplink, installation the other way around with small amount of throughput
						if (uplink) {
							// IPv4 instance of the AP
							addr = this->_nodesAP.Get(i)
								->GetObject<Ipv4>()
								->GetAddress(1, 0) // Loopback (1-0)
								.GetLocal();

							clientCT.SetAttribute ("RemoteAddress",AddressValue(addr));
							clientCT.SetAttribute ("RemotePort",UintegerValue(port));
							clientCT.SetAttribute ("MaxPackets", UintegerValue(1e9));
							clientCT.SetAttribute ("Interval", TimeValue(Seconds(15.0 * this->_intervals[this->_associations[i][j]])));
							clientCT.SetAttribute ("PacketSize", UintegerValue(this->_packetSize / 8.0));

							apps = clientCT.Install(this->_nodesSta[i].Get(j));
							apps.Start(Seconds(applicationStart));
							apps.Stop(Seconds(applicationEnd));
						}
				}
		}

		Simulator::Stop(Seconds(applicationEnd+0.01));

		// Optimization relative objects
		// Init callback for configuration changes
		switch (e) {
			case DEF: this->_entryPoints = {defaultConf}; break;
			case DEGA:
				this->_entryPoints = this->findDegreeEntryPoints();
				std::vector<NetworkConfiguration> confs_nh = this->findNHDegreeEntryPoints(),
																					def = {defaultConf};
				this->_entryPoints.insert(this->_entryPoints.end(), confs_nh.begin(), confs_nh.end());
				this->_entryPoints.insert(this->_entryPoints.end(), def.begin(), def.end());				
				break;
		}

		// for (NetworkConfiguration nc: this->_entryPoints) {
		// 	for (std::tuple<double, double> t: nc) std::cout << "(" << std::get<0>(t) << "," << std::get<1>(t) << "),";
		// 	std::cout << std::endl;
		// }

		if (sId != HCM && sId != HGM)
			this->_testCounter = this->_entryPoints.size();
		unsigned int nParams = std::set<unsigned int>(this->_clustersAP.begin(), this->_clustersAP.end()).size();
		this->_configuration = this->_entryPoints[0];

		this->setupNewConfiguration(this->_configuration);
		Simulator::Schedule(Seconds(applicationStart+testDuration), &Simulation::endOfTest, this);

		// Init containers for throughput calculation
		for (int i = 0; i < numberOfAPs; i++) {
			unsigned int nStas = this->_associations[i].size();
			this->_throughputs.push_back(std::vector<double>(nStas, 0));
			this->_pers.push_back(std::vector<double>(nStas, 0));
			this->_lastRxPackets.push_back(std::vector<unsigned int>(nStas, 0));
			this->_lastLostPackets.push_back(std::vector<unsigned int>(nStas, 0));
		}

		// Parameters to optimize
		std::vector<ConstrainedCouple> parameters(nParams);
		for (unsigned int i = 0; i < nParams; i++) {
			parameters[i] = ConstrainedCouple(SingleParameter(-82, -62, 1), SingleParameter(1, 21, 1), &parameterConstraint);
		}

		// Sampler to use
		Sampler* sampler = nullptr;
		switch (sId) {
			case UNIF: sampler = new UniformSampler(parameters); break;
			case HGM: sampler = new HGMTSampler(parameters, this->_entryPoints, 6, 1.0 / (numberOfStas + 1.0), 1.0, &numberOfSamples); break;
			case HCM: sampler = new HCMSampler(parameters, this->_entryPoints, 6, 1.0 / (numberOfStas + 1.0), &numberOfSamples_HCM, dmode); break;
		}

		// Optimizer to use
		switch (oId) {
			case IDLEOPT: this->_optimizer = new IdleOptimizer(); break;
			case EGREEDY: this->_optimizer = new EpsilonGreedyOptimizer(sampler, 0.1); break;
			case THOMP_GAMNORM: this->_optimizer = new ThompsonGammaNormalOptimizer(sampler, 3, 0.1, 0, sId != HGM); break;
			case THOMP_NORM: this->_optimizer = new ThompsonNormalOptimizer(sampler, 0.1); break;
			case MARGIN: this->_optimizer = nullptr; break;
			case RANDNEIGHBOR: this->_optimizer = new RandomNeighborOptimizer(sampler); break;
		}

		std::cout << "ns3-debug: the simulation begins" << std::endl;
		Simulator::Run();

		Simulator::Destroy();

		// Stringstream for vector data
		std::ofstream myfile;
		myfile.open ("./scratch/nsTest/data/" + outputName);
		myfile << "rew,fair,cum,aps,stas,conf" << std::endl;
		for (unsigned int i = 0; i < this->_rewards.size(); i++) {
			std::stringstream aps, stas, pers;
			std::string delimiter = ",";
			copy(this->_apThroughputs[i].begin(), this->_apThroughputs[i].end(), std::ostream_iterator<double>(aps, delimiter.c_str()));
			copy(this->_staThroughputs[i].begin(), this->_staThroughputs[i].end(), std::ostream_iterator<double>(stas, delimiter.c_str()));
			copy(this->_staPERs[i].begin(), this->_staPERs[i].end(), std::ostream_iterator<double>(pers, delimiter.c_str()));

			std::string apsData = aps.str();
			std::string stasData = stas.str();
			std::string persData = pers.str();
			apsData = apsData.substr(0, apsData.size() - 1);
			stasData = stasData.substr(0, stasData.size() - 1);
			persData = persData.substr(0, persData.size() - 1);

			myfile << this->_rewards[i] << "\t" << this->_fairness[i] << "\t" << this->_cumulatedThroughput[i] << "\t"
						 << apsData << "\t" << stasData << "\t" << persData << "\t"
						 << this->configurationToString(this->_configurations[i]) << std::endl;
		}
		myfile.close();

		// Free sampler and optimizer
		delete sampler;
		if (this->_optimizer != nullptr)
			delete this->_optimizer;
		
		exit(0);
	}
}

/**
 * Return the PID of the simulation
 * 
 * @return the PID of the simulation
 */
pid_t Simulation::getPID() const {
	return this->_pid;
}

/**
 * Compute the adequate reward from throughputs of STAs
 * 
 * @return the adequate reward computation
 */
double Simulation::rewardFromThroughputs() {
	std::vector<std::vector<double>> attainables = this->attainableThroughputs();
	switch (this->_rewardType) {
		case AD_HOC: return this->adHocReward(this->_throughputs, attainables); break;
		case CUMTP: return this->cumulatedThroughputReward(this->_throughputs, attainables); break;
		case LOGPF: return this->logPfReward(this->_throughputs, attainables); break;
	}

	return -1;
}

/**
 * Compute the adequate reward from throughputs of STAs
 * 
 * @return the adequate reward computation
 */
double Simulation::rewardFromThroughputs(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) {
	switch (this->_rewardType) {
		case AD_HOC: return this->adHocReward(throughputs, attainables); break;
		case CUMTP: return this->cumulatedThroughputReward(throughputs, attainables); break;
		case LOGPF: return this->logPfReward(throughputs, attainables); break;
	}

	return -1;
}

/**
 * This function is called at each end of test
 */
void Simulation::endOfTest() {
	this->_testCounter++;

  // Compute the throughput
  this->computeThroughputsAndErrors();

	if (!this->_warmed && this->_testCounter >= this->_warmup_tests) {
		this->_warmed = true;
		this->_testCounter = 1;
	}

	if (this->_warmed) {
		// Store metrics
		this->storeMetrics();
		// Compute reward accordingly
		double rew = this->rewardFromThroughputs();
		std::vector<std::tuple<double, unsigned int>> subrews = this->subrewardFromThroughputs();

		double alpha = 2.0 / 31.0;
		if (this->_ema < 0.0) this->_ema = rew;
		else this->_ema = alpha * rew + (1 - alpha) * this->_ema;
		this->_cumulative += rew;

		std::cout << "Reward at t = " << this->_testCounter * this->_testDuration << ": " << rew << " (Cum: " << this->_cumulative << ", EMA: " << this->_ema << ")" << std::endl << std::endl;

		// Add config and reward to sampler
		NetworkConfiguration configuration = this->_configuration;
		if (this->_optimizer != nullptr) {
			this->_optimizer->addToBase(this->_configuration, rew, true, subrews);

			// Use the optimizer to get another configuration
			if (this->_testCounter / this->_optimizer->getTestPeriod() < this->_entryPoints.size() && this->_optimizer->readyForAnother()) {
				configuration = this->_entryPoints[this->_testCounter / this->_optimizer->getTestPeriod()];
			} else {
				configuration = this->_optimizer->optimize();
			}
		} else if (!this->_changed) {
			this->_changed = true;
			configuration = {};
			double margin = 5;
			std::vector<double> rssis = this->getLogDistanceRSSIs();
			double obsspd_min = -82, obsspd_max = -62,
					tx_min = 1, tx_max = 21;
			for (double rssi: rssis) {
				double sens = std::max(obsspd_min, std::min(obsspd_max, round(rssi - margin)));
				double tx = std::max(tx_min, std::min(tx_max, -62 - sens));
				configuration.push_back(std::make_tuple(sens, tx));
			}
		}

		for (std::tuple<double, double> t: configuration) {
			std::cout << "(" << std::get<0>(t) << ", " << std::get<1>(t) << "), ";
		}
		std::cout << std::endl;

		// Set up the new configuration
		setupNewConfiguration(configuration);
	}

  // Next scheduling for recurrent callback
  Simulator::Schedule(Seconds(this->_testDuration), &Simulation::endOfTest, this);
}

/**
 * Build the network configuration according to the topo clusters
 * 
 * @param configuration NetworkConfiguration the network configuration clusterized
 * 
 * @return the unclusterized network configuration
 */
NetworkConfiguration Simulation::handleClusterizedConfiguration(const NetworkConfiguration& configuration) {
	unsigned int numberAps = this->_positionAPX.size();
	NetworkConfiguration unclusterized(numberAps);
	unsigned int k = 0;
	for (std::tuple<double, double> couple: configuration) {
		for (unsigned int i = 0; i < numberAps; i++) {
			if (this->_clustersAP[i] == k) {
				unclusterized[i] = couple;
			}
		}
		k++;
	}

	return unclusterized;
}

/**
 * Store the network metrics in dedicated containers.
 * This method should be called AFTER computeThroughputsAndErrors.
 */
void Simulation::storeMetrics() {
	this->_configurations.push_back(this->handleClusterizedConfiguration(this->_configuration));

	double rew = this->rewardFromThroughputs();
	this->_rewards.push_back(rew);
	// std::cout << "Reward: " << this->rewardFromThroughputs() << " vs. " << this->otherRewardFromThroughputs() << std::endl;

	double fairness = this->fairnessFromThroughputs();
	this->_fairness.push_back(fairness);
	// std::cout << "Fairness: " << fairness << std::endl;

	double cumThrough = this->cumulatedThroughputFromThroughputs();
	this->_cumulatedThroughput.push_back(cumThrough);
	// std::cout << "CumThrough: " << cumThrough << std::endl << std::endl;

	this->_apThroughputs.push_back(this->apThroughputsFromThroughputs());
	this->_staThroughputs.push_back(this->staThroughputsFromThroughputs());
	this->_staPERs.push_back(this->staPersFromPers());
}

std::vector<std::vector<double>> Simulation::attainableThroughputs() {
	std::vector<std::vector<double>> attainables;
	std::vector<double> references = Simulation::attainableThroughputsFromChannel(this->_channel_width);
	unsigned int i = 0;
	for (NodeContainer::Iterator iap = this->_nodesAP.Begin(); iap != this->_nodesAP.End(); iap++, i++) {
    Node* ap = GetPointer(*iap);
		std::vector<double> attainablesSta;
		unsigned int j = 0;
		for (NodeContainer::Iterator istation = this->_nodesSta[i].Begin(); istation != this->_nodesSta[i].End(); istation++, j++) {
			unsigned int sta_idx = this->_associations[i][j];
			Node* sta = GetPointer(*istation);
			unsigned int mcsValue = Simulation::getMCSValue(ap, sta);
			attainablesSta.push_back(std::max(1.0, std::min(this->_packetSize / this->_intervals[sta_idx], references[mcsValue])));
		}

		attainables.push_back(attainablesSta);
  }

	return attainables;
}

std::vector<double> Simulation::attainableThroughputsFromChannel(ChannelWidth cw) {
	std::vector<double> at;
	switch (cw) {
		case MHZ_20: return {30e6, 55e6, 85e6, 106e6, 165e6, 215e6, 240e6, 265e6, 315e6, 345e6, 390e6, 430e6};
		case MHZ_40: return {55e6, 110e6, 165e6, 215e6, 320e6, 400e6, 455e6, 500e6, 580e6, 645e6, 700e6, 785e6};
		case MHZ_80: return {120e6, 230e6, 330e6, 430e6, 610e6, 780e6, 850e6, 890e6, 1070e6, 1180e6, 1220e6, 1400e6};
	}

	return {};
}

double Simulation::adHocReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) {
	double starvRew = 1, noStarvRew = 1, nStarv = 0, nNoStarv = 0;
  for (unsigned int i = 0; i < throughputs.size(); i++) {
		double n = throughputs[i].size();
		for (unsigned int j = 0; j < n; j++) {
      double threshold = 0.1 * attainables[i][j] / n;
      if (throughputs[i][j] < threshold) {
        throughputs[i][j] = std::max(throughputs[i][j], 1.0);
        starvRew *= throughputs[i][j] / threshold;
				nStarv++;
      } else {
				noStarvRew *= throughputs[i][j] / attainables[i][j];
				nNoStarv++;
			}
    }
	}
		
  double n = nStarv + nNoStarv;
  // Compute global reward
  return (nStarv * starvRew + nNoStarv * (noStarvRew + n)) / (n * (n + 1.0));
}

std::vector<std::tuple<double, unsigned int>> Simulation::subrewardFromThroughputs() {
	std::vector<std::vector<double>> attainables = this->attainableThroughputs();
	std::vector<std::tuple<double, unsigned int>> subrewards;

	for (unsigned int i = 0; i < this->_throughputs.size(); i++) {
		std::tuple<double, unsigned int> t;
		switch (this->_rewardType) {
			case AD_HOC: t = std::make_tuple(this->adHocReward({this->_throughputs[i]}, {attainables[i]}), this->_throughputs[i].size()); break;
			case CUMTP: t = std::make_tuple(this->cumulatedThroughputReward({this->_throughputs[i]}, {attainables[i]}), this->_throughputs[i].size()); break;
			case LOGPF: t = std::make_tuple(this->logPfReward({this->_throughputs[i]}, {attainables[i]}), this->_throughputs[i].size()); break;
		}
		subrewards.push_back(t);
	}

	return subrewards;
}

double Simulation::logPfReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) {
	double logT = 0.0, logTA = 0.0;
  for (unsigned int i = 0; i < throughputs.size(); i++) {
		double n = throughputs[i].size();
		for (unsigned int j = 0; j < n; j++) {
			logT += log(std::max(1.0, throughputs[i][j]));
			logTA += log(attainables[i][j]);
    }
	}
  // Compute global reward
  return logT / logTA;
}

/**
 * Compute the fairness of the network (Jain's index).
 * 
 * @return the fairness 
 */
double Simulation::fairnessFromThroughputs() {
  double squareOfMean = 0, meanOfSquares = 0;
	unsigned int n = 0;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
    for (unsigned int j = 0; j < this->_throughputs[i].size(); j++) {
			double tMbps = this->_throughputs[i][j] / 1.0e6;
      squareOfMean += tMbps;
			meanOfSquares += tMbps * tMbps;
			n++;
    }

	squareOfMean *= squareOfMean / (n * n);
	meanOfSquares /= n;

	return squareOfMean / meanOfSquares;
}

/**
 * Compute the cumulated throughput.
 * 
 * @return the cumulated throughput 
 */
double Simulation::cumulatedThroughputReward(std::vector<std::vector<double>> throughputs, std::vector<std::vector<double>> attainables) {
  double cumThroughput = 0;
  for (unsigned int i = 0; i < throughputs.size(); i++)
    for (unsigned int j = 0; j < throughputs[i].size(); j++)
			cumThroughput += throughputs[i][j];

	return cumThroughput;
}

/**
 * Compute the cumulated throughput.
 * 
 * @return the cumulated throughput 
 */
double Simulation::cumulatedThroughputFromThroughputs() {
  double cumThroughput = 0;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
    for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			cumThroughput += this->_throughputs[i][j];

	return cumThroughput;
}

/**
 * Compute the throughput of each AP.
 * 
 * @return the throughput of each AP 
 */
std::vector<double> Simulation::apThroughputsFromThroughputs() {
  std::vector<double> apThroughputs;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++) {
		double apThroughput = 0;
		for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			apThroughput += this->_throughputs[i][j];
		apThroughputs.push_back(apThroughput);
	}

	return apThroughputs;
}

/**
 * Compute the throughput of each STA.
 * 
 * @return the throughput of each STA 
 */
std::vector<double> Simulation::staThroughputsFromThroughputs() {
  std::vector<double> staThroughputs;
  for (unsigned int i = 0; i < this->_throughputs.size(); i++)
		for (unsigned int j = 0; j < this->_throughputs[i].size(); j++)
			staThroughputs.push_back(this->_throughputs[i][j]);

	return staThroughputs;
}

/**
 * Compute the PER of each STA.
 * 
 * @return the PER of each STA 
 */
std::vector<double> Simulation::staPersFromPers() {
  std::vector<double> staPers;
  for (unsigned int i = 0; i < this->_pers.size(); i++)
		for (unsigned int j = 0; j < this->_pers[i].size(); j++)
			staPers.push_back(this->_pers[i][j]);

	return staPers;
}

/**
 * Compute the throughput of each station
 */
void Simulation::computeThroughputsAndErrors() {
  // Compute throughput for each server app
  for (unsigned int i = 0; i < this->_serversPerAp.size(); i++) {
    std::vector<double> staAPThroughputs(this->_lastRxPackets[i].size());
		std::vector<double> staPERs(this->_lastLostPackets[i].size());
    for (unsigned int j = 0; j < this->_serversPerAp[i].GetN(); j++) {
      // Received bytes since the start of the simulation
			UdpServer* server = dynamic_cast<UdpServer*>(GetPointer(this->_serversPerAp[i].Get(j)));
			unsigned int lostPackets = server->GetLost();
			unsigned int testLostPackets = lostPackets - this->_lastLostPackets[i][j];
			unsigned int receivedPackets = server->GetReceived();
			unsigned int testReceivedPackets = receivedPackets - this->_lastRxPackets[i][j];
			double per = testLostPackets + testReceivedPackets != 0 ? ((double) testLostPackets) / (testLostPackets + testReceivedPackets) : 0;

			// std::cout << testLostPackets << " and " << testReceivedPackets << " => " << per << std::endl;

      double rxBits = this->_packetSize * testReceivedPackets;
      // Compute the throughput considering only unseen bytes
      double throughput = rxBits / this->_testDuration; // bit/s
			// std::cout << i << " " << j << ": " << throughput << std::endl;
      // Update containers
      staAPThroughputs[j] = throughput;
			staPERs[j] = per;
      this->_lastRxPackets[i][j] = receivedPackets;
			this->_lastLostPackets[i][j] = lostPackets;
      // Log
      // std::cout << "Station " << j << " of AP " << i << " : " << (throughput / 1e6) << " Mbit/s" << std::endl;
    }
    this->_throughputs[i] = staAPThroughputs;
		this->_pers[i] = staPERs;
  }
}

/**
 * Set up a new configuration in the simulation
 * 
 * @param configuration the configuration to set
 */ 
void Simulation::setupNewConfiguration(NetworkConfiguration configuration) {
  int nNodes = this->_devices.size();
	NetworkConfiguration unclusterized = this->handleClusterizedConfiguration(configuration);
  for (int i = 0; i < nNodes; i++) {
    Ptr<WifiPhy> phy = dynamic_cast<WifiNetDevice*>(GetPointer((this->_devices[i].Get(0))))->GetPhy();

    double sensibility = std::get<0>(unclusterized[i]),
           txPower = std::get<1>(unclusterized[i]);
    phy->SetTxPowerEnd(txPower);
    phy->SetTxPowerStart(txPower);
    phy->SetRxSensitivity(sensibility);
    phy->SetCcaEdThreshold(sensibility);
  }

	this->_configuration = configuration;
}

/**
 * Extract topology information from its JSON representation
 * 
 * @param topo JSON representation of the topology
 */
void Simulation::readTopology(Json::Value topo) {  
  // APs
	unsigned int c = 0;
	bool clusterized = false;
	if (topo["aps"][0].isMember("cluster")) clusterized = true;

  for (Json::Value ap: topo["aps"]) {
    this->_positionAPX.push_back(ap["x"].asDouble());
    this->_positionAPY.push_back(ap["y"].asDouble());
    this->_positionAPZ.push_back(ap["z"].asDouble());

		if (clusterized) {
			this->_clustersAP.push_back(ap["cluster"].asUInt64());
		} else {
			this->_clustersAP.push_back(c);
			c++;
		}

    std::vector<unsigned int> assoc;
    for (Json::Value sta: ap["stas"]) {
      assoc.push_back(sta.asUInt());
    }
    this->_associations.push_back(assoc);
  }

  // Stations
  for (Json::Value sta: topo["stations"]) {
    this->_positionStaX.push_back(sta["x"].asDouble());
    this->_positionStaY.push_back(sta["y"].asDouble());
    this->_positionStaZ.push_back(sta["z"].asDouble());
  }
}

/**
 * Boolean constraint for parameter combination
 * 
 * @param sens double the sensibility
 * @param pow double the transmission power
 * 
 * @return true if the constraint is validated, false otherwise 
 */
bool Simulation::parameterConstraint(double sens, double pow) {
  return sens <= std::max(-82.0, std::min(-62.0, -82.0 + 20.0 - pow));
}

/**
 * Compute the number of samples for a given gaussian mixture
 * 
 * @param gaussians Container the gaussian mixture
 * 
 * @return the number of tests to do before updating the mixture
 */
unsigned int Simulation::numberOfSamples(std::vector<GaussianT> gaussians) {
  double s = 0;
  for (GaussianT g: gaussians) {
		s += 0.5 * 2.0 * std::get<0>(g).size() * std::get<1>(g) / 0.05;
	}

  return round(s);
}

void Simulation::stationsThroughputsToInterval(const std::vector<StationThroughput>& stations_throughputs, double duration) {
	this->_intervals.clear();
	for (StationThroughput stt: stations_throughputs) {
		double interval;
		switch (stt) {
			case NONE: interval = duration; break;
			case LOW: interval = this->_packetSize / 500e3; break;
			case MEDIUM: interval = this->_packetSize / 5e6; break;
			case HIGH: interval = this->_packetSize / 50e6; break;
		}
		this->_intervals.push_back(interval);
	}
}

/**
 * Compute the number of samples for a given gaussian mixture
 * 
 * @param gaussians Container the gaussian mixture
 * 
 * @return the number of tests to do before updating the mixture
 */
unsigned int Simulation::numberOfSamples_HCM(std::vector<Ring> circulars) {
  return 2 * circulars.size();
}

/**
 * Map channel index to a real channel.
 *
 * Channel number must be in {36, 40, etc.} for the 5GHz band.
 * The web page: https://www.nsnam.org/docs/models/html/wifi-user.html
 * Read in particular the WifiPhy::ChannelNumber section
 *
 * @param i int the index to map
 *
 * @return a channel corresponding to the mapped index
 */
int Simulation::channelNumber(ChannelWidth cw) {
  switch (cw) {
    case MHZ_20: return 36;
    case MHZ_40: return 38;
    case MHZ_80: return 42;
    default:
      std::cerr << "Error channelNumber(): the index is negative, null, or greater than the number of channels (12 - 40MHz)." << std::endl;
      return 42;
  }
}

/**
 * Turn a network configuration to a convenient string representation
 * 
 * @param config Container the configuration
 * 
 * @return a convenient representation of the configuration
 */
std::string Simulation::configurationToString(const NetworkConfiguration& config) {
	std::string result = "";
	for (unsigned int i = 0; i < config.size(); i++) {
		result += "(" + std::to_string(std::get<0>(config[i])) + "," + std::to_string(std::get<1>(config[i])) + ")";
		if (i < config.size() - 1)
			result += ",";
	}

	return result;
}

std::vector<NetworkConfiguration> Simulation::findDegreeEntryPoints(double criterion) const {
	int tx = 20;
	double avg_deg = 1000;
	unsigned int n = this->_positionAPX.size();
	NetworkConfiguration conf;
	do {
		conf = NetworkConfiguration(n, std::make_tuple(-62 - tx, tx));
		std::vector<std::vector<unsigned int>> conflicts = this->extractConflicts(conf);
		unsigned int sum = 0;
		for (std::vector<unsigned int> c: conflicts) {
			sum += c.size();
		}
		avg_deg = ((double) sum) / n;
		tx--;
	} while (avg_deg > criterion);

	return {conf};
}

std::vector<NetworkConfiguration> Simulation::findNHDegreeEntryPoints(double criterion) const {
	double avg_deg = 1000;
	unsigned int n = this->_positionAPX.size();
	NetworkConfiguration conf(n, std::make_tuple(-82, 20));
	do {
		std::vector<std::vector<unsigned int>> conflicts = this->extractConflicts(conf);
		unsigned int max_idx = 0;
		unsigned int sum = conflicts[0].size();
		for (unsigned int i = 1; i < conflicts.size(); i++) {
			unsigned int nconflicts = conflicts[i].size();
			if (nconflicts > conflicts[max_idx].size())
				max_idx = i;
			sum += nconflicts;
		}

		avg_deg = ((double) sum) / n;

		if (avg_deg < criterion)
			break;
		
		std::get<1>(conf[max_idx]) = std::get<1>(conf[max_idx]) - 1;
		std::get<0>(conf[max_idx]) = -62 - std::get<1>(conf[max_idx]);
	} while (avg_deg > criterion);

	return {conf};
}

std::vector<NetworkConfiguration> Simulation::findDiagonalEntryPoints(unsigned int n) const {
	unsigned int conf_size = this->_positionAPX.size();
	std::default_random_engine generator(std::chrono::system_clock::now().time_since_epoch().count());
	std::vector<NetworkConfiguration> confs;
	for (unsigned int i = 0; i < n; i++) {
		NetworkConfiguration conf;
		for (unsigned int j = 0; j < conf_size; j++) {
			double max_dist = 0;
			for (unsigned int sId: this->_associations[j]) {
				double d = Simulation::distance(std::make_tuple(this->_positionAPX[j], this->_positionAPY[j], this->_positionAPZ[j]), std::make_tuple(this->_positionStaX[sId], this->_positionStaY[sId], this->_positionStaZ[sId]));
				if (d > max_dist)
					max_dist = d;
			}
			unsigned int minTx = std::min(std::max(1.0, ceil(-82.0 + 46.67 + 10 * 3 * log10(max_dist))), 21.0);
			std::uniform_int_distribution<int> dist(minTx, 21);
			int tx = dist(generator);
			conf.push_back(std::make_tuple(-62 - tx, tx));
		}
		confs.push_back(conf);
	}

	return confs;
}

std::vector<NetworkConfiguration> Simulation::findEntryPoints(int v) const {
	unsigned int n = this->_positionAPX.size();
	std::vector<unsigned int> order(n, 0);
	for (unsigned int i = 0; i < n; i++) order[i] = i;
	NetworkConfiguration def_conf(n, std::make_tuple(-82, 20)),
											 conf = def_conf,
											 prev_conf = conf;
	std::vector<std::vector<unsigned int>> conflicts = extractConflicts(conf);
	do {
		prev_conf = conf;
		std::random_shuffle(order.begin(), order.end());

		for (unsigned int k: order) {
			std::vector<double> rxs;
			for (unsigned int l: conflicts[k]) {
				rxs.push_back(Simulation::pathLoss(std::make_tuple(this->_positionAPX[k], this->_positionAPY[k], this->_positionAPZ[k]), std::make_tuple(this->_positionAPX[l], this->_positionAPY[l], this->_positionAPZ[l]), std::get<1>(conf[l])));
			}

			if (!rxs.empty()) {
				std::vector<double>::iterator elem;
				if (v == -1 || v >= (int) rxs.size()) {
					elem = std::min_element(rxs.begin(), rxs.end());
				} else {
					std::nth_element(rxs.begin(), rxs.begin() + v - 1, rxs.end(), std::greater<double>());
					elem = rxs.begin() + v - 1;
				}
				double new_sens = std::max(std::min(ceil(*elem - 1), -62.0), -82.0);
				double new_tx = -62 - new_sens;
				conf[k] = std::make_tuple(new_sens, new_tx);
			}
		}
		// for (std::tuple<double, double> t: conf) {
		// 	std::cout << "(" << std::get<0>(t) << "," << std::get<1>(t) << ")";
		// }
		// std::cout << std::endl;
	} while (conf != prev_conf);

	return {conf};
}

std::vector<std::vector<unsigned int>> Simulation::extractConflicts(NetworkConfiguration conf) const {
	std::vector<std::vector<unsigned int>> conflicts;
	for (unsigned int i = 0; i < this->_positionAPX.size(); i++) {
		std::vector<unsigned int> conflictsi;
		for (unsigned int j = 0; j < this->_positionAPX.size(); j++) {
			if (i != j) {
				double rx = Simulation::pathLoss(std::make_tuple(this->_positionAPX[i], this->_positionAPY[i], this->_positionAPZ[i]), std::make_tuple(this->_positionAPX[j], this->_positionAPY[j], this->_positionAPZ[j]), std::get<1>(conf[i]));
				if (rx >= std::get<0>(conf[j]))
					conflictsi.push_back(j);
			}
		}

		conflicts.push_back(conflictsi);
	}

	return conflicts;
}

std::vector<double> Simulation::getLogDistanceRSSIs() const {
	std::vector<double> rssis;
	for (unsigned int i = 0; i < this->_positionAPX.size(); i++) {
		double rssi = -200;
		for (unsigned int j = 0; j < this->_positionAPX.size(); j++) {
			if (i != j) {
				double rx = Simulation::pathLoss(std::make_tuple(this->_positionAPX[i], this->_positionAPY[i], this->_positionAPZ[i]), std::make_tuple(this->_positionAPX[j], this->_positionAPY[j], this->_positionAPZ[j]), std::get<1>(this->_configuration[i]));
				if (rssi < rx)
					rssi = rx;
			}
		}
		rssis.push_back(rssi);
	}

	return rssis;
}

double Simulation::pathLoss(std::tuple<double, double, double> source, std::tuple<double, double, double> target, double txPower) {
	double d = Simulation::distance(source, target);
	return txPower - 46.67 - 10 * 3 * log10(d);
}

double Simulation::distance(std::tuple<double, double, double> source, std::tuple<double, double, double> target) {
	return sqrt(pow(std::get<0>(source) - std::get<0>(target), 2) + pow(std::get<1>(source) - std::get<1>(target), 2) + pow(std::get<2>(source) - std::get<2>(target), 2));
}

WifiNetDevice* Simulation::getWifiDevice(Node* node) {
	return dynamic_cast<WifiNetDevice*>(GetPointer(node->GetDevice(0)));
}

WifiMac* Simulation::getMAC(Node* node) {
	return GetPointer(getWifiDevice(node)->GetMac());
}

WifiMacHeader Simulation::createAdHocMacHeader(Node* from, Node* to) {
	WifiMac* macFrom = getMAC(from), *macTo = getMAC(to);
	WifiMacHeader wmh;
	wmh.SetAddr1(macTo->GetAddress());
	wmh.SetAddr2(macFrom->GetBssid());
	wmh.SetAddr3(macFrom->GetAddress());
	wmh.SetType(WIFI_MAC_DATA);
	wmh.SetDsFrom();

	return wmh;
}

WifiMode Simulation::getWifiMode(Node* from, Node* to) {
	WifiNetDevice* from_wnd = getWifiDevice(from);
	WifiMacHeader adhoc = createAdHocMacHeader(from, to);
	return GetPointer(from_wnd->GetRemoteStationManager())->GetDataTxVector(adhoc).GetMode();
}

unsigned int Simulation::getMCSValue(Node* from, Node* to) {
	WifiMode wm = getWifiMode(from, to);
	return wm.GetMcsValue();
}

std::string Simulation::getMCSClass(Node* from, Node* to) {
	WifiMode wm = getWifiMode(from, to);
	std::string modClass = "";
	switch (wm.GetModulationClass()) {
		case WIFI_MOD_CLASS_HT: modClass = "Ht"; break;
		case WIFI_MOD_CLASS_VHT: modClass = "Vht"; break;
		case WIFI_MOD_CLASS_HE: modClass = "He"; break;
		default: break;
	}

	return modClass;
}