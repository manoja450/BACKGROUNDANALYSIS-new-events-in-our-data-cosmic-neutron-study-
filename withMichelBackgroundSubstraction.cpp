#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TH1D.h>
#include <TH2D.h>
#include <TF1.h>
#include <TCanvas.h>
#include <TSystem.h>
#include <TMath.h>
#include <TStyle.h>
#include <TLegend.h>
#include <TPaveStats.h>
#include <TGaxis.h>
#include <TPaveText.h>
#include <TBox.h>
#include <TLatex.h>
#include <TParameter.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <string>
#include <map>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <ctime>
#include <TGraph.h>
#include <TGraphErrors.h>
#include <TPad.h>

using std::cout;
using std::cerr;
using std::endl;
using namespace std;

// Constants
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const int PULSE_THRESHOLD = 30;     // ADC threshold for pulse detection
const int BS_UNCERTAINTY = 5;       // Baseline uncertainty (ADC)
const int EV61_THRESHOLD = 1200;    // Beam on if channel 22 > this (ADC)
const double MUON_ENERGY_THRESHOLD = 50; // Min PMT energy for muon (p.e.)
const double MICHEL_ENERGY_MIN = 0;    // Min PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX = 1000;  // Max PMT energy for Michel (p.e.)
const double MICHEL_ENERGY_MAX_DT = 500; // Max PMT energy for dt plots (p.e.)
const double MICHEL_DT_MIN = 0.76;      // Min time after muon for Michel (µs)
const double MICHEL_DT_MAX = 16.0;      // Max time after muon for Michel (µs)
const double MICHEL_DT_MIN_EXTENDED = 0.0;      // Min time for extended Michel search (µs)
const double MICHEL_DT_MAX_EXTENDED = 16.0;     // Max time for extended Michel search (µs)
const double MICHEL_ENERGY_MAX_EXTENDED = 100.0; // Max energy for extended Michel (p.e.)
const int ADCSIZE = 45;                 // Number of ADC samples per waveform
const double LOW_ENERGY_DT_MIN = 16.0;  // Min time after muon for low-energy isolated events (µs)
const std::vector<double> SIDE_VP_THRESHOLDS = {750, 950, 1200, 1400, 550, 700, 700, 500}; // Channels 12-19 (ADC)
const double TOP_VP_THRESHOLD = 450; // Original threshold
const double FIT_MIN = 1.0; // Fit range min for Michel dt (µs)
const double FIT_MAX = 16.0; // Fit range max for Michel dt (µs)
const double FIT_MIN_LOW_MUON = 16.0; // Fit range min for low to muon dt (µs)
const double FIT_MAX_LOW_MUON = 1200.0; // Fit range max for low to muon dt (µs, narrowed)

// Michel background prediction constants
const double SIGNAL_REGION_MIN = 16.0;    // Signal region start (µs)
const double SIGNAL_REGION_MAX = 100.0;   // Signal region end (µs)

// Generate unique output directory with timestamp
string getTimestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", t);
    return string(buffer);
}
const string OUTPUT_DIR = "./AnalysisOutput_" + getTimestamp();

// Pulse structure
struct pulse {
    double start;          // Start time (µs)
    double end;            // End time (µs)
    double peak;           // Max amplitude (p.e. for PMTs, ADC for SiPMs)
    double energy;         // Energy (p.e. for PMTs, ADC for SiPMs)
    double number;         // Number of channels with pulse
    bool single;           // Timing consistency
    bool beam;             // Beam status
    int trigger;           // Trigger type
    double side_vp_energy; // Side veto energy (ADC)
    double top_vp_energy;  // Top veto energy (ADC)
    double all_vp_energy;  // All veto energy (ADC)
    double last_muon_time; // Time of last muon (µs)
    bool is_muon;          // Muon candidate flag
    bool is_michel;        // Michel electron candidate flag
    bool veto_hit[10];     // Which veto panels were hit (channels 12-21)
};

// Temporary pulse structure
struct pulse_temp {
    double start;  // Start time (µs)
    double end;    // End time (µs)
    double peak;   // Max amplitude
    double energy; // Energy
};

// SPE fitting function
Double_t SPEfit(Double_t *x, Double_t *par) {
    Double_t term1 = par[0] * exp(-0.5 * pow((x[0] - par[1]) / par[2], 2));
    Double_t term2 = par[3] * exp(-0.5 * pow((x[0] - par[4]) / par[5], 2));
    Double_t term3 = par[6] * exp(-0.5 * pow((x[0] - sqrt(2) * par[4]) / sqrt(2 * pow(par[5], 2) - pow(par[2], 2)), 2));
    Double_t term4 = par[7] * exp(-0.5 * pow((x[0] - sqrt(3) * par[4]) / sqrt(3 * pow(par[5], 2) - 2 * pow(par[2], 2)), 2));
    return term1 + term2 + term3 + term4;
}

// Exponential fit function
Double_t ExpFit(Double_t *x, Double_t *par) {
    return par[0] * exp(-x[0] / par[1]) + par[2];
}

// Utility functions
template<typename T>
double getAverage(const std::vector<T>& v) {
    if (v.empty()) return 0;
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

template<typename T>
double mostFrequent(const std::vector<T>& v) {
    if (v.empty()) return 0;
    std::map<T, int> count;
    for (const auto& val : v) count[val]++;
    T most_common = v[0];
    int max_count = 0;
    for (const auto& pair : count) {
        if (pair.second > max_count) {
            max_count = pair.second;
            most_common = pair.first;
        }
    }
    return max_count > 1 ? most_common : getAverage(v);
}

template<typename T>
double variance(const std::vector<T>& v) {
    if (v.size() <= 1) return 0;
    double mean = getAverage(v);
    double sum = 0;
    for (const auto& val : v) {
        sum += (val - mean) * (val - mean);
    }
    return sum / (v.size() - 1);
}

// Create output directory
void createOutputDirectory(const string& dirName) {
    struct stat st;
    if (stat(dirName.c_str(), &st) != 0) {
        if (mkdir(dirName.c_str(), 0755) != 0) {
            cerr << "Error: Could not create directory " << dirName << endl;
            exit(1);
        }
        cout << "Created output directory: " << dirName << endl;
    } else {
        cout << "Output directory already exists: " << dirName << endl;
    }
}

// SPE calibration function
void performCalibration(const string &calibFileName, Double_t *mu1, Double_t *mu1_err) {
    TFile *calibFile = TFile::Open(calibFileName.c_str());
    if (!calibFile || calibFile->IsZombie()) {
        cerr << "Error opening calibration file: " << calibFileName << endl;
        exit(1);
    }

    TTree *calibTree = (TTree*)calibFile->Get("tree");
    if (!calibTree) {
        cerr << "Error accessing tree in calibration file" << endl;
        calibFile->Close();
        exit(1);
    }

    TCanvas *c = new TCanvas("c", "SPE Fits", 1200, 800);
    TH1F *histArea[N_PMTS];
    Long64_t nLEDFlashes[N_PMTS] = {0};
    for (int i = 0; i < N_PMTS; i++) {
        histArea[i] = new TH1F(Form("PMT%d_Area", i + 1),
                               Form("PMT %d;ADC Counts;Events", i + 1), 150, -50, 400);
    }

    Int_t triggerBits;
    Double_t area[23];
    calibTree->SetBranchAddress("triggerBits", &triggerBits);
    calibTree->SetBranchAddress("area", area);

    Long64_t nEntries = calibTree->GetEntries();
    cout << "Processing " << nEntries << " calibration events from " << calibFileName << "..." << endl;

    for (Long64_t entry = 0; entry < nEntries; entry++) {
        calibTree->GetEntry(entry);
        if (triggerBits != 16) continue;
        for (int pmt = 0; pmt < N_PMTS; pmt++) {
            histArea[pmt]->Fill(area[PMT_CHANNEL_MAP[pmt]]);
            nLEDFlashes[pmt]++;
        }
    }

    for (int i = 0; i < N_PMTS; i++) {
        if (histArea[i]->GetEntries() < 1000) {
            cerr << "Warning: Insufficient data for PMT " << i + 1 << " in " << calibFileName << endl;
            mu1[i] = 0;
            mu1_err[i] = 0;
            delete histArea[i];
            continue;
        }

        TF1 *fitFunc = new TF1("fitFunc", SPEfit, -50, 400, 8);
        Double_t histMean = histArea[i]->GetMean();
        Double_t histRMS = histArea[i]->GetRMS();

        fitFunc->SetParameters(1000, histMean - histRMS, histRMS / 2,
                              1000, histMean, histRMS,
                              500, 200);

        histArea[i]->Fit(fitFunc, "Q", "", -50, 400);

        mu1[i] = fitFunc->GetParameter(4);
        Double_t sigma_mu1 = fitFunc->GetParError(4);
        Double_t sigma1 = fitFunc->GetParameter(5);
        mu1_err[i] = sqrt(pow(sigma_mu1, 2) + pow(sigma1 / sqrt(nLEDFlashes[i]), 2));

        c->Clear();
        histArea[i]->Draw();
        fitFunc->Draw("same");
        TLegend *leg = new TLegend(0.6, 0.7, 0.9, 0.9);
        leg->AddEntry(histArea[i], Form("PMT %d Data", i + 1), "l");
        leg->AddEntry(fitFunc, "SPE Fit", "l");
        leg->AddEntry((TObject*)0, Form("mu1 = %.2f #pm %.2f", mu1[i], mu1_err[i]), "");
        leg->Draw();
        string plotName = OUTPUT_DIR + Form("/SPE_Fit_PMT%d.png", i + 1);
        c->Update();
        c->SaveAs(plotName.c_str());
        cout << "Saved SPE plot: " << plotName << endl;
        delete leg;
        delete fitFunc;
        delete histArea[i];
    }

    delete c;
    calibFile->Close();
}

void createVetoPanelPlots(TH1D* h_veto_panel[10], const string& outputDir) {
    for (int i = 0; i < 10; i++) {
        TCanvas *c = new TCanvas(Form("c_veto_%d", i+12), Form("Veto Panel %d", i+12), 1200, 800);
        gStyle->SetOptStat(1111);
        gStyle->SetOptTitle(1);
        gStyle->SetStatX(0.9);
        gStyle->SetStatY(0.9);
        gStyle->SetStatW(0.2);
        gStyle->SetStatH(0.15);
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(2);
        h_veto_panel[i]->Draw("hist");
        string plotName = outputDir + Form("/Veto_Panel_%d.png", i+12);
        c->SaveAs(plotName.c_str());
        cout << "Saved veto panel plot: " << plotName << endl;
        delete c;
    }

    TCanvas *c_combined = new TCanvas("c_veto_combined", "Combined Veto Panel Energies", 1600, 1200);
    c_combined->Divide(4, 3);
    for (int i = 0; i < 10; i++) {
        c_combined->cd(i+1);
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(2);
        h_veto_panel[i]->SetTitle("");
        h_veto_panel[i]->Draw("hist");
    }
    string combinedPlotName = outputDir + "/Combined_Veto_Panels.png";
    c_combined->SaveAs(combinedPlotName.c_str());
    cout << "Saved combined veto panel plot: " << combinedPlotName << endl;
    delete c_combined;
}

// Function to save cosmic ray counts to a single CSV file
void saveCosmicFluxToCSV(const std::map<Long64_t, int>& cosmic_counts, const string& outputDir) {
    string filename = outputDir + "/CosmicFlux_AllHours.csv";
    ofstream csv_file(filename);
    csv_file << "DateTime,EventCount\n";
    for (const auto& pair : cosmic_counts) {
        Long64_t hour_start = pair.first;
        int count = pair.second;
        struct tm *timeinfo = localtime((time_t*)&hour_start);
        char buffer[20];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", timeinfo);
        csv_file << buffer << "," << count << "\n";
    }
    csv_file.close();
    cout << "Saved cosmic flux data: " << filename << endl;
}

// Function to save daily cosmic ray averages to a CSV file
void saveDailyCosmicFluxToCSV(const std::vector<std::pair<Long64_t, int>>& file_infos, const string& outputDir) {
    string filename = outputDir + "/CosmicFlux_AllDays.csv";
    ofstream csv_file(filename);
    if (!csv_file) {
        cerr << "Error: Could not open file " << filename << endl;
        return;
    }
    csv_file << "Date,DailyCount\n";
    size_t num = file_infos.size();
    for (size_t i = 0; i < num; i += 24) {
        int sum = 0;
        size_t group_size = 0;
        Long64_t first_ts = file_infos[i].first;
        for (size_t j = 0; j < 24 && i + j < num; j++) {
            sum += file_infos[i + j].second;
            group_size++;
        }
        double avg = static_cast<double>(sum) / group_size;
        time_t now = static_cast<time_t>(first_ts);
        struct tm *t = localtime(&now);
        char buffer[11];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d", t);
        csv_file << buffer << "," << avg << "\n";
    }
    csv_file.close();
    cout << "Saved daily cosmic flux data: " << filename << endl;
}

// Function to calculate total events in histogram by summing bin contents
double calculateTotalEvents(TH1D* hist) {
    double total = 0;
    for (int i = 1; i <= hist->GetNbinsX(); i++) {
        total += hist->GetBinContent(i);
    }
    return total;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <calibration_file> <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }

    string calibFileName = argv[1];
    vector<string> inputFiles;
    for (int i = 2; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }

    // Sort input files by starttime
    std::vector<std::pair<Long64_t, string>> file_times;
    for (const auto& file : inputFiles) {
        TFile f_tmp(file.c_str());
        if (f_tmp.IsZombie()) {
            cerr << "Warning: Cannot open " << file << " for timestamp sorting, skipping" << endl;
            f_tmp.Close();
            continue;
        }
        auto tsstart = (TParameter<Long64_t>*)f_tmp.Get("starttime");
        Long64_t ts = tsstart ? tsstart->GetVal() : 0;
        f_tmp.Close();
        file_times.emplace_back(ts, file);
    }
    std::sort(file_times.begin(), file_times.end());
    inputFiles.clear();
    for (const auto& ft : file_times) inputFiles.push_back(ft.second);

    createOutputDirectory(OUTPUT_DIR);

    cout << "Calibration file: " << calibFileName << endl;
    cout << "Input files (sorted by starttime):\n";
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    if (gSystem->AccessPathName(calibFileName.c_str())) {
        cerr << "Error: Calibration file " << calibFileName << " not found" << endl;
        return -1;
    }

    bool anyInputFileExists = false;
    for (const auto& file : inputFiles) {
        if (!gSystem->AccessPathName(file.c_str())) {
            anyInputFileExists = true;
            break;
        }
    }
    if (!anyInputFileExists) {
        cerr << "Error: No input files found" << endl;
        return -1;
    }

    // Perform SPE calibration
    Double_t mu1[N_PMTS] = {0};
    Double_t mu1_err[N_PMTS] = {0};
    performCalibration(calibFileName, mu1, mu1_err);

    cout << "SPE Calibration Results (from " << calibFileName << "):\n";
    for (int i = 0; i < N_PMTS; i++) {
        cout << "PMT " << i + 1 << ": mu1 = " << mu1[i] << " ± " << mu1_err[i] << " ADC counts/p.e.\n";
    }

    // Statistics counters
    int num_muons = 0;
    int num_michels = 0;
    int num_michels_extended = 0;
    int num_events = 0;
    int num_cosmic_events = 0;
    int num_low_iso = 0;

    std::map<int, int> trigger_counts;

    // Define histograms
    TH1D* h_muon_energy = new TH1D("muon_energy", "Muon Energy Distribution (with Michel Electrons);Energy (p.e.);Counts/7 p.e.", 500, -500, 3000);
    TH1D* h_muon_all = new TH1D("muon_all", "All Muon Energy Distribution;Energy (p.e.);Counts/6 p.e.", 500, 0, 3000);
    TH1D* h_michel_energy = new TH1D("michel_energy", "Michel Electron Energy Distribution;Energy (p.e.);Counts/8 p.e.", 100, 0, 800);
    TH1D* h_dt_michel = new TH1D("DeltaT", "Muon-Michel Time Difference;Time to Previous event(Muon)(#mus);Counts/0.08 #mus", 200, 0, MICHEL_DT_MAX);
    TH2D* h_energy_vs_dt = new TH2D("energy_vs_dt", "Michel Energy vs Time Difference;dt (#mus);Energy (p.e.)", 160, 0, 16, 200, 0, 1000);
    TH1D* h_side_vp_muon = new TH1D("side_vp_muon", "Side Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 5000);
    TH1D* h_top_vp_muon = new TH1D("top_vp_muon", "Top Veto Energy for Muons;Energy (ADC);Counts", 200, 0, 1000);
    TH1D* h_trigger_bits = new TH1D("trigger_bits", "Trigger Bits Distribution;Trigger Bits;Counts", 36, 0, 36);
    TH1D* h_isolated_pe = new TH1D("isolated_pe", "Sum PEs Isolated Events;Photoelectrons;Counts/10 p.e.", 200, 0, 2000);
    TH1D* h_low_iso = new TH1D("low_iso", "Sum PEs Low Energy Isolated Events;Photoelectrons;Counts/1 p.e.", 100, 0, 100);
    TH1D* h_high_iso = new TH1D("high_iso", "Sum PEs High Energy Isolated Events;Photoelectrons;Counts/10 p.e.", 100, 0, 1000);
    TH1D* h_dt_prompt_delayed = new TH1D("dt_prompt_delayed", "#Delta t High Energy (prompt) to Low Energy (delayed);#Delta t [#mus];Counts", 200, 0, 10000);
    TH1D* h_dt_low_muon = new TH1D("dt_low_muon", "#Delta t Low Energy Isolated to Muon Veto Tagged Events;#Delta t [#mus];Counts/10#mus", 120, 0, 1200);
    TH1D* h_dt_high_muon = new TH1D("dt_high_muon", "#Delta t High Energy Isolated to Muon Veto Tagged Events;#Delta t [#mus];Counts/10#mus", 120, 0, 1200);
    TH1D* h_low_pe_signal = new TH1D("low_pe_signal", "Low Energy Signal Region Sideband Subtraction;Photoelectrons;Counts", 100, 0, 100);
    TH1D* h_low_pe_sideband = new TH1D("low_pe_sideband", "Low Energy Sideband (1000-1200 #mus);Photoelectrons;Counts", 100, 0, 100);
    TH1D* h_isolated_ge40 = new TH1D("isolated_ge40", "Sum PEs Isolated Events (>=40 p.e.);Photoelectrons;Events/20 p.e.", 200, 40, 2000);

    // Extended Michel analysis histograms for background subtraction
    TH1D* h_dt_michel_sideband = new TH1D("dt_michel_sideband", 
        "Michel Time Distribution (0-16 #mus);Time to Previous Muon (#mus);Counts", 
        80, 0, 16);
    TH1D* h_michel_energy_sideband = new TH1D("michel_energy_sideband", 
        "Michel Energy Spectrum;Energy (p.e.);Counts", 
        100, 0, 100);

    // Histograms for veto panels (12-21)
    TH1D* h_veto_panel[10];
    const char* veto_names[10] = {
        "Veto Panel 12", "Veto Panel 13", "Veto Panel 14", "Veto Panel 15",
        "Veto Panel 16", "Veto Panel 17", "Veto Panel 18", "Veto Panel 19",
        "Veto Panel 20", "Veto Panel 21"
    };

    for (int i = 0; i < 10; i++) {
        h_veto_panel[i] = new TH1D(Form("h_veto_panel_%d", i+12), 
                                   Form("%s;Energy (ADC);Counts", veto_names[i]), 
                                   200, 0, 8000);
    }

    // Neutron Purity Analysis histograms with larger axis titles
    TH1D* h_neutron_richness = new TH1D("neutron_richness", 
        "Neutron-to-Background Ratio vs Time;Time[#mus];Neutron/Bkg Ratio", 
        100, 0, 1000);
    TH1D* h_signal_significance = new TH1D("signal_significance", 
        "Signal Significance vs Time;Time[#mus];S/#sqrt{S + B}", 
        100, 0, 1000);

    // For Multi-dimensional analysis
    TH2D* h_energy_vs_time_low = new TH2D("energy_vs_time_low", 
        "Low Energy Events: Energy vs Time to Muon;Time to Muon [#mus];Energy (p.e.)", 
        120, 0, 1200, 100, 0, 100);
    TH2D* h_energy_vs_time_high = new TH2D("energy_vs_time_high", 
        "High Energy Events: Energy vs Time to Muon;Time to Muon [#mus];Energy (p.e.)", 
        120, 0, 1200, 200, 0, 2000);

    // Declare last times outside the loop to carry over across files
    double last_muon_time = 0.0;
    double last_high_time = 0.0;

    // Global list for muon candidates across all files
    std::vector<std::pair<double, double>> muon_candidates;
    std::set<double> michel_muon_times;

    // Global cosmic counts across all files (keyed by hour start timestamp)
    std::map<Long64_t, int> cosmic_counts;

    // Vector to store per-file cosmic info for daily calculation
    std::vector<std::pair<Long64_t, int>> file_cosmic_infos;

    // Set of excluded trigger bits for cosmic events
    const std::set<int> excluded_triggers = {1, 3, 4, 8, 16, 33, 35};

    // Counter for excluded low-energy isolated events
    int excluded_low_iso = 0;

    for (const auto& inputFileName : inputFiles) {
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = new TFile(inputFileName.c_str());
        cout << "Processing file: " << inputFileName << endl;

        // Get run start time
        Long64_t run_starttime = 0;
        auto tsstart = (TParameter<Long64_t> *) f->Get("starttime");
        if (tsstart) {
            run_starttime = tsstart->GetVal();
            cout << "Run Start Time (Unix Timestamp): " << run_starttime << endl;
            time_t rawtime = (time_t)run_starttime;
            struct tm *timeinfo = localtime(&rawtime);
            cout << "Run Start Time (Local Time): " << asctime(timeinfo);
            timeinfo->tm_min = 0;
            timeinfo->tm_sec = 0;
            run_starttime = mktime(timeinfo);
        } else {
            cerr << "Warning: 'starttime' not found in file " << inputFileName << ". Skipping file." << endl;
            f->Close();
            continue;
        }

        TTree* t = (TTree*)f->Get("tree");
        if (!t) {
            cout << "Could not find tree in file: " << inputFileName << endl;
            f->Close();
            continue;
        }

        // Declaration of leaf types
        Int_t eventID;
        Int_t nSamples[23];
        Short_t adcVal[23][45];
        Double_t baselineMean[23];
        Double_t baselineRMS[23];
        Double_t pulseH[23];
        Int_t peakPosition[23];
        Double_t area[23];
        Long64_t nsTime;
        Int_t triggerBits;

        t->SetBranchAddress("eventID", &eventID);
        t->SetBranchAddress("nSamples", nSamples);
        t->SetBranchAddress("adcVal", adcVal);
        t->SetBranchAddress("baselineMean", baselineMean);
        t->SetBranchAddress("baselineRMS", baselineRMS);
        t->SetBranchAddress("pulseH", pulseH);
        t->SetBranchAddress("peakPosition", peakPosition);
        t->SetBranchAddress("area", area);
        t->SetBranchAddress("nsTime", &nsTime);
        t->SetBranchAddress("triggerBits", &triggerBits);

        int numEntries = t->GetEntries();
        cout << "Processing " << numEntries << " entries in " << inputFileName << endl;

        int file_cosmic_count = 0;
        double last_event_time = -1;

        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;

            // Check for non-monotonic event times
            if (nsTime < last_event_time) {
                cerr << "Warning: Non-increasing event time in file " << inputFileName 
                     << ", event " << eventID << ": " << nsTime << " < " << last_event_time << endl;
            }
            last_event_time = nsTime;

            h_trigger_bits->Fill(triggerBits);
            trigger_counts[triggerBits]++;
            if (triggerBits < 0 || triggerBits > 36) {
                cout << "Warning: triggerBits = " << triggerBits << " out of histogram range (0-36) in file " << inputFileName << ", event " << eventID << endl;
            }

            struct pulse p;
            p.start = nsTime / 1000.0;
            p.end = p.start;
            p.peak = 0;
            p.energy = 0;
            p.number = 0;
            p.single = false;
            p.beam = false;
            p.trigger = triggerBits;
            p.side_vp_energy = 0;
            p.top_vp_energy = 0;
            p.all_vp_energy = 0;
            p.last_muon_time = last_muon_time;
            p.is_muon = false;
            p.is_michel = false;
            for (int i = 0; i < 10; i++) p.veto_hit[i] = false;

            std::vector<double> all_chan_start, all_chan_end, all_chan_peak, all_chan_energy;
            std::vector<double> side_vp_energy, top_vp_energy;
            std::vector<double> chan_starts_no_outliers;
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);

            bool pulse_at_end = false;
            int pulse_at_end_count = 0;
            std::vector<double> veto_energies(10, 0);

            for (int iChan = 0; iChan < 23; iChan++) {
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                if (iChan == 22) {
                    double ev61_energy = 0;
                    for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                        ev61_energy += h_wf.GetBinContent(iBin);
                    }
                    if (ev61_energy > EV61_THRESHOLD) {
                        p.beam = true;
                    }
                }

                std::vector<pulse_temp> pulses_temp;
                bool onPulse = false;
                int thresholdBin = 0, peakBin = 0;
                double peak = 0, pulseEnergy = 0;
                double allPulseEnergy = 0;

                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    double iBinContent = h_wf.GetBinContent(iBin);
                    if (iBin > 15) allPulseEnergy += iBinContent;

                    if (!onPulse && iBinContent >= PULSE_THRESHOLD) {
                        onPulse = true;
                        thresholdBin = iBin;
                        peakBin = iBin;
                        peak = iBinContent;
                        pulseEnergy = iBinContent;
                    } else if (onPulse) {
                        pulseEnergy += iBinContent;
                        if (peak < iBinContent) {
                            peak = iBinContent;
                            peakBin = iBin;
                        }
                        if (iBinContent < BS_UNCERTAINTY || iBin == ADCSIZE) {
                            pulse_temp pt;
                            pt.start = thresholdBin * 16.0 / 1000.0;
                            pt.peak = iChan <= 11 && mu1[iChan] > 0 ? peak / mu1[iChan] : peak;
                            pt.end = iBin * 16.0 / 1000.0;
                            for (int j = peakBin - 1; j >= 1 && h_wf.GetBinContent(j) > BS_UNCERTAINTY; j--) {
                                if (h_wf.GetBinContent(j) > peak * 0.1) {
                                    pt.start = j * 16.0 / 1000.0;
                                }
                                pulseEnergy += h_wf.GetBinContent(j);
                            }
                            if (iChan <= 11) {
                                pt.energy = mu1[iChan] > 0 ? pulseEnergy / mu1[iChan] : 0;
                                all_chan_start.push_back(pt.start);
                                all_chan_end.push_back(pt.end);
                                all_chan_peak.push_back(pt.peak);
                                all_chan_energy.push_back(pt.energy);
                                if (pt.energy > 1) p.number += 1;
                            }
                            pulses_temp.push_back(pt);
                            peak = 0;
                            pulseEnergy = 0;
                            thresholdBin = 0;
                            onPulse = false;
                        }
                    }
                }

                if (iChan >= 12 && iChan <= 19) {
                    side_vp_energy.push_back(allPulseEnergy);
                    veto_energies[iChan - 12] = allPulseEnergy;
                    if (allPulseEnergy > SIDE_VP_THRESHOLDS[iChan - 12]) {
                        p.veto_hit[iChan - 12] = true;
                    }
                } else if (iChan >= 20 && iChan <= 21) {
                    double factor = (iChan == 20) ? 1.07809 : 1.0;
                    top_vp_energy.push_back(allPulseEnergy * factor);
                    veto_energies[iChan - 12] = allPulseEnergy * factor;
                    if (allPulseEnergy * factor > TOP_VP_THRESHOLD) {
                        p.veto_hit[iChan - 12] = true;
                    }
                }

                if (iChan <= 11 && h_wf.GetBinContent(ADCSIZE) > 100) {
                    pulse_at_end_count++;
                    if (pulse_at_end_count >= 10) pulse_at_end = true;
                }

                h_wf.Reset();
            }

            p.start += mostFrequent(all_chan_start);
            p.end += mostFrequent(all_chan_end);
            p.energy = std::accumulate(all_chan_energy.begin(), all_chan_energy.end(), 0.0);
            p.peak = std::accumulate(all_chan_peak.begin(), all_chan_peak.end(), 0.0);
            p.side_vp_energy = std::accumulate(side_vp_energy.begin(), side_vp_energy.end(), 0.0);
            p.top_vp_energy = std::accumulate(top_vp_energy.begin(), top_vp_energy.end(), 0.0);
            p.all_vp_energy = p.side_vp_energy + p.top_vp_energy;

            for (const auto& start : all_chan_start) {
                if (fabs(start - mostFrequent(all_chan_start)) < 10 * 16.0 / 1000.0) {
                    chan_starts_no_outliers.push_back(start);
                }
            }
            p.single = (variance(chan_starts_no_outliers) < 5 * 16.0 / 1000.0);

            bool veto_hit = false;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_hit = true;
                    break;
                }
            }
            if (!veto_hit && p.top_vp_energy > TOP_VP_THRESHOLD) veto_hit = true;

            bool veto_low = !veto_hit;

            bool is_cosmic = !p.beam && excluded_triggers.find(p.trigger) == excluded_triggers.end();
            if (is_cosmic) {
                num_cosmic_events++;
                file_cosmic_count++;
            }

            if ((p.energy > MUON_ENERGY_THRESHOLD && veto_hit) ||
                (pulse_at_end && p.energy > MUON_ENERGY_THRESHOLD / 2 && veto_hit)) {
                p.is_muon = true;
                last_muon_time = p.start;
                num_muons++;
                muon_candidates.emplace_back(p.start, p.energy);
                h_muon_all->Fill(p.energy);
                h_side_vp_muon->Fill(p.side_vp_energy);
                h_top_vp_muon->Fill(p.top_vp_energy);
                
                for (int i = 0; i < 10; i++) {
                    if (p.veto_hit[i]) {
                        h_veto_panel[i]->Fill(veto_energies[i]);
                    }
                }
            }

            bool is_beam_off = ((p.trigger & 1) == 0) && (p.trigger != 4) && (p.trigger != 8) && (p.trigger != 16) && !p.beam;

            double dt = p.start - last_muon_time;

            // ORIGINAL MICHEL ANALYSIS (0-16 μs) - UNCHANGED
            bool is_michel_candidate = is_beam_off &&
                                      p.energy >= MICHEL_ENERGY_MIN &&
                                      p.energy <= MICHEL_ENERGY_MAX &&
                                      dt >= MICHEL_DT_MIN &&
                                      dt <= MICHEL_DT_MAX &&
                                      p.number >= 8 &&
                                      veto_low;
            h_energy_vs_dt->Fill(dt, p.energy);

            bool is_michel_for_dt = is_michel_candidate && p.energy <= MICHEL_ENERGY_MAX_DT;

            if (is_michel_candidate) {
                p.is_michel = true;
                num_michels++;
                michel_muon_times.insert(last_muon_time);
                h_michel_energy->Fill(p.energy);
            }

            if (is_michel_for_dt) {
                h_dt_michel->Fill(dt);
            }

            // EXTENDED MICHEL ANALYSIS FOR BACKGROUND SUBTRACTION (0-16 μs sideband)
            bool is_michel_sideband = is_beam_off &&
                                     p.energy >= MICHEL_ENERGY_MIN &&
                                     p.energy <= MICHEL_ENERGY_MAX_EXTENDED &&
                                     dt >= MICHEL_DT_MIN_EXTENDED &&
                                     dt <= MICHEL_DT_MAX_EXTENDED &&
                                     p.number >= 8 &&
                                     veto_low;

            if (is_michel_sideband) {
                num_michels_extended++;
                h_dt_michel_sideband->Fill(dt);
                h_michel_energy_sideband->Fill(p.energy);
            }

            p.last_muon_time = last_muon_time;

            bool is_isolated = is_beam_off &&
                               !p.is_muon &&
                               !p.is_michel &&
                               veto_low &&
                               p.single &&
                               p.energy > 0;

            if (is_isolated) {
                double dt_muon = p.start - last_muon_time;
                
                // Fill multi-dimensional histograms
                if (p.energy <= 100 && p.number >= 4) {
                    h_energy_vs_time_low->Fill(dt_muon, p.energy);
                } else if (p.energy > 100 && p.number >= 8) {
                    h_energy_vs_time_high->Fill(dt_muon, p.energy);
                }
                
                if (p.energy > 100 && p.number >= 8) {
                    h_high_iso->Fill(p.energy);
                    h_isolated_pe->Fill(p.energy);
                    h_isolated_ge40->Fill(p.energy);
                    last_high_time = p.start;
                    if (dt_muon >= LOW_ENERGY_DT_MIN) {
                        h_dt_high_muon->Fill(dt_muon);
                    }
                } else if (p.energy <= 100 && p.number >= 4) {
                    num_low_iso++;
                    double dt_high = p.start - last_high_time;
    
                    // Fill h_dt_prompt_delayed for all low-energy isolated events
                    if (dt_high >= 0 && dt_high <= 10000) {
                        h_dt_prompt_delayed->Fill(dt_high);
                    } else {
                        h_dt_prompt_delayed->Fill(10000); // Overflow bin for invalid dt_high
                        excluded_low_iso++;
                    }
                    
                    h_low_iso->Fill(p.energy);
                    if (p.energy >= 40) {
                        h_isolated_ge40->Fill(p.energy);
                    }
                    if (dt_muon >= LOW_ENERGY_DT_MIN) {
                        h_dt_low_muon->Fill(dt_muon);
                    }
                    // CORRECTED: Neutron-rich region is now 16-100 μs
                    if (dt_muon > 16 && dt_muon < 100) {
                        h_low_pe_signal->Fill(p.energy);
                    }
                    if (dt_muon > 1000 && dt_muon < 1200) {
                        h_low_pe_sideband->Fill(p.energy);
                    }
                }
            }
        }

        if (file_cosmic_count > 0) {
            cosmic_counts[run_starttime] += file_cosmic_count;
        }

        file_cosmic_infos.push_back({run_starttime, file_cosmic_count});

        f->Close();
        delete f;
    }

    if (!cosmic_counts.empty()) {
        saveCosmicFluxToCSV(cosmic_counts, OUTPUT_DIR);
    } else {
        cerr << "Error: No valid cosmic events or starttime found in any input file. Skipping CSV output." << endl;
    }

    if (!file_cosmic_infos.empty()) {
        saveDailyCosmicFluxToCSV(file_cosmic_infos, OUTPUT_DIR);
    } else {
        cerr << "Error: No file cosmic info available for daily CSV." << endl;
    }

    for (const auto& muon : muon_candidates) {
        if (michel_muon_times.find(muon.first) != michel_muon_times.end()) {
            h_muon_energy->Fill(muon.second);
        }
    }

    // Verify histogram entries
    cout << "h_low_iso entries: " << h_low_iso->GetEntries() << endl;
    cout << "h_dt_prompt_delayed entries: " << h_dt_prompt_delayed->GetEntries() << endl;
    cout << "num_low_iso: " << num_low_iso << endl;
    cout << "Excluded low-energy isolated events from h_dt_prompt_delayed (dt_high < 0 or > 10000 µs): " << excluded_low_iso << endl;

    cout << "Global Statistics:\n";
    cout << "Total Events: " << num_events << "\n";
    cout << "Muons Detected: " << num_muons << "\n";
    cout << "Michel Electrons Detected (0-16 μs): " << num_michels << "\n";
    cout << "Michel Events in Sideband (0-16 μs): " << num_michels_extended << "\n";
    cout << "Low-Energy Isolated Events: " << num_low_iso << "\n";
    cout << "Prompt-Delayed Pairs (h_dt_prompt_delayed entries): " << h_dt_prompt_delayed->GetEntries() << "\n";
    cout << "Cosmic Ray Events Detected: " << num_cosmic_events << "\n";
    cout << "------------------------\n";

    cout << "Trigger Bits Distribution (all files):\n";
    for (const auto& pair : trigger_counts) {
        cout << "Trigger " << pair.first << ": " << pair.second << " events\n";
    }
    cout << "------------------------\n";

    TCanvas *c = new TCanvas("c", "Analysis Plots", 1200, 800);
    gStyle->SetOptStat(1111);
    gStyle->SetOptFit(1111);

    // ==== ALL ORIGINAL PLOTS - RETAINED ====
    
    c->Clear();
    h_muon_energy->SetLineColor(kBlue);
    h_muon_energy->Draw();
    c->Update();
    string plotName = OUTPUT_DIR + "/Muon_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_muon_all->SetLineColor(kBlue);
    h_muon_all->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Muon_All_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_michel_energy->SetLineColor(kRed);
    h_michel_energy->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_dt_michel->SetLineWidth(2);
    h_dt_michel->SetLineColor(kBlack);
    h_dt_michel->GetXaxis()->SetTitle("Time to previous event (Muon) [#mus]");
    h_dt_michel->Draw("HIST");

    TF1* expFit = nullptr;
    if (h_dt_michel->GetEntries() > 5) {
        double integral = h_dt_michel->Integral(h_dt_michel->FindBin(FIT_MIN), h_dt_michel->FindBin(FIT_MAX));
        double bin_width = h_dt_michel->GetBinWidth(1);
        double N0_init = integral * bin_width / (FIT_MAX - FIT_MIN);
        double C_init = 0;
        int bin_12 = h_dt_michel->FindBin(12.0);
        int bin_16 = h_dt_michel->FindBin(16.0);
        double min_content = 1e9;
        for (int i = bin_12; i <= bin_16; i++) {
            double content = h_dt_michel->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1;

        expFit = new TF1("expFit", ExpFit, FIT_MIN, FIT_MAX, 3);
        expFit->SetParameters(N0_init, 2.2, C_init);
        expFit->SetParLimits(0, 0, N0_init * 100);
        expFit->SetParLimits(1, 0.1, 20.0);
        expFit->SetParLimits(2, -C_init * 10, C_init * 10);
        expFit->SetParNames("N_{0}", "#tau", "C");
        expFit->SetLineColor(kRed);
        expFit->SetLineWidth(3);

        int fitStatus = h_dt_michel->Fit(expFit, "RE+", "SAME", FIT_MIN, FIT_MAX);
        expFit->Draw("SAME");

        gPad->Update();
        TPaveStats *stats = (TPaveStats*)h_dt_michel->FindObject("stats");
        if (stats) {
            stats->SetX1NDC(0.6);
            stats->SetX2NDC(0.9);
            stats->SetY1NDC(0.6);
            stats->SetY2NDC(0.9);
            stats->SetTextColor(kRed);
        }

        double N0 = expFit->GetParameter(0);
        double N0_err = expFit->GetParError(0);
        double tau = expFit->GetParameter(1);
        double tau_err = expFit->GetParError(1);
        double C = expFit->GetParameter(2);
        double C_err = expFit->GetParError(2);
        double chi2 = expFit->GetChisquare();
        int ndf = expFit->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Exponential Fit Results (Michel dt, " << FIT_MIN << "-" << FIT_MAX << " µs):\n";
        cout << "Fit Status: " << fitStatus << " (0 = success)\n";
        cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
        cout << Form("N₀ = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;
    } else {
        cout << "Warning: h_dt_michel has insufficient entries (" << h_dt_michel->GetEntries() 
             << "), skipping exponential fit" << endl;
    }

    c->Update();
    plotName = OUTPUT_DIR + "/Michel_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    if (expFit) delete expFit;

    // Fit start comparison plot
    if (h_dt_michel->GetEntries() > 5) {
        h_dt_michel->GetListOfFunctions()->Clear();
        std::vector<double> fit_starts = {1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0};
        std::vector<double> taus, tau_errs, chi2ndfs;
        int best_index = -1;
        double min_chi2ndf = 1e9;

        for (int i = 0; i < fit_starts.size(); i++) {
            double fit_start = fit_starts[i];
            double fit_end = 16.0;

            TF1* expFit_var = new TF1(Form("expFit_var_%.1f", fit_start), ExpFit, fit_start, fit_end, 3);

            double C_init = 0;
            int bin_12 = h_dt_michel->FindBin(12.0);
            int bin_16 = h_dt_michel->FindBin(16.0);
            double min_content = 1e9;
            for (int bin = bin_12; bin <= bin_16; bin++) {
                double content = h_dt_michel->GetBinContent(bin);
                if (content > 0 && content < min_content) min_content = content;
            }
            if (min_content < 1e9) C_init = min_content;
            else C_init = 0.1;

            double integral = h_dt_michel->Integral(h_dt_michel->FindBin(fit_start), h_dt_michel->FindBin(fit_end));
            double bin_width = h_dt_michel->GetBinWidth(1);
            double N0_init = (integral * bin_width - C_init * (fit_end - fit_start)) / 2.2;
            if (N0_init < 0) N0_init = 100;

            expFit_var->SetParameters(N0_init, 2.2, C_init);
            expFit_var->SetParNames("N_{0}", "#tau", "C");
            expFit_var->SetParLimits(0, 0, N0_init * 100);
            expFit_var->SetParLimits(1, 0.1, 20.0);
            expFit_var->SetParLimits(2, -C_init * 10, C_init * 10);

            int fitStatus = h_dt_michel->Fit(expFit_var, "QRN+", "", fit_start, fit_end);

            double tau = expFit_var->GetParameter(1);
            double tau_err = expFit_var->GetParError(1);
            double chi2 = expFit_var->GetChisquare();
            int ndf = expFit_var->GetNDF();
            double chi2ndf = (ndf > 0) ? chi2 / ndf : 999;

            taus.push_back(tau);
            tau_errs.push_back(tau_err);
            chi2ndfs.push_back(chi2ndf);

            if (chi2ndf < min_chi2ndf && fitStatus == 0) {
                min_chi2ndf = chi2ndf;
                best_index = i;
            }

            cout << Form("Fit Range %.1f-%.1f µs:\n", fit_start, fit_end);
            cout << "Fit Status: " << fitStatus << " (0 = success)\n";
            cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
            cout << Form("χ²/NDF = %.4f", chi2ndf) << endl;
            cout << "----------------------------------------" << endl;

            delete expFit_var;
        }

        if (best_index >= 0) {
            cout << Form("Best Fit Range: %.1f-16.0 µs\n", fit_starts[best_index]);
            cout << Form("τ = %.4f ± %.4f µs", taus[best_index], tau_errs[best_index]) << endl;
            cout << Form("χ²/NDF = %.4f (minimum)", chi2ndfs[best_index]) << endl;
            cout << "----------------------------------------" << endl;
        }

        TCanvas* c_comp = new TCanvas("c_comp", "Fit Start Time Comparison", 1200, 800);
        c_comp->SetGrid();

        TPad* pad = new TPad("pad", "pad", 0, 0, 1, 1);
        pad->Draw();
        pad->cd();

        TGraph* g_chi2 = new TGraph(fit_starts.size(), &fit_starts[0], &chi2ndfs[0]);
        TGraph* g_tau = new TGraph(fit_starts.size(), &fit_starts[0], &taus[0]);

        g_chi2->SetTitle("Fit Start Time Comparison");
        g_chi2->GetXaxis()->SetTitle("Fit Start Time (#mus)");
        g_chi2->GetYaxis()->SetTitle("#chi^{2}/ndf");
        g_chi2->SetMarkerStyle(20);
        g_chi2->SetMarkerColor(kBlue);
        g_chi2->SetLineColor(kBlue);
        g_chi2->SetLineWidth(2);

        g_tau->SetMarkerStyle(22);
        g_tau->SetMarkerColor(kRed);
        g_tau->SetLineColor(kRed);
        g_tau->SetLineWidth(2);

        g_chi2->Draw("APL");

        pad->Update();
        double ymin = pad->GetUymin();
        double ymax = pad->GetUymax();

        double tau_min = *std::min_element(taus.begin(), taus.end());
        double tau_max = *std::max_element(taus.begin(), taus.end());
        double scale = (ymax - ymin)/(tau_max - tau_min);
        double offset = ymin - tau_min * scale;

        for (int i = 0; i < g_tau->GetN(); i++) {
            double x, y;
            g_tau->GetPoint(i, x, y);
            g_tau->SetPoint(i, x, y * scale + offset);
        }

        g_tau->Draw("PL same");

        TGaxis* axis = new TGaxis(gPad->GetUxmax(), gPad->GetUymin(),
                                 gPad->GetUxmax(), gPad->GetUymax(),
                                 tau_min, tau_max, 510, "+L");
        axis->SetLineColor(kRed);
        axis->SetLabelColor(kRed);
        axis->SetTitle("#tau (#mus)");
        axis->SetTitleColor(kRed);
        axis->Draw();

        TLegend* leg = new TLegend(0.7, 0.7, 0.9, 0.9);
        leg->AddEntry(g_chi2, "#chi^{2}/ndf", "lp");
        leg->AddEntry(g_tau, "#tau", "lp");
        leg->Draw();

        string compPlotName = OUTPUT_DIR + "/FitStartComparison.png";
        c_comp->SaveAs(compPlotName.c_str());
        cout << "Saved comparison plot: " << compPlotName << endl;

        delete g_chi2;
        delete g_tau;
        delete leg;
        delete axis;
        delete pad;
        delete c_comp;
    } else {
        cout << "Skipping fit start comparison - insufficient entries in dt histogram" << endl;
    }

    c->Clear();
    h_energy_vs_dt->SetStats(0);
    h_energy_vs_dt->GetXaxis()->SetTitle("dt (#mus)");
    h_energy_vs_dt->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Michel_Energy_vs_dt.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_side_vp_muon->SetLineColor(kMagenta);
    h_side_vp_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Side_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_top_vp_muon->SetLineColor(kCyan);
    h_top_vp_muon->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Top_Veto_Muon.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_trigger_bits->SetLineColor(kGreen);
    h_trigger_bits->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/TriggerBits_Distribution.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    createVetoPanelPlots(h_veto_panel, OUTPUT_DIR);

    c->Clear();
    h_isolated_pe->SetLineColor(kBlack);
    h_isolated_pe->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Isolated_PE.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_low_iso->SetLineColor(kBlack);
    h_low_iso->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Low_Energy_Isolated.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_high_iso->SetLineColor(kBlack);
    h_high_iso->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/High_Energy_Isolated.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    c->Clear();
    h_dt_prompt_delayed->SetLineColor(kBlack);
    h_dt_prompt_delayed->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/DeltaT_High_to_Low.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    TCanvas *c_low_muon = new TCanvas("c_low_muon", "DeltaT Low to Muon", 1200, 800);
    c_low_muon->SetLeftMargin(0.15);
    c_low_muon->SetRightMargin(0.08);
    c_low_muon->SetBottomMargin(0.12);
    c_low_muon->SetTopMargin(0.08);

    h_dt_low_muon->SetLineWidth(2);
    h_dt_low_muon->SetLineColor(kBlue);
    h_dt_low_muon->SetFillColor(kBlue);
    h_dt_low_muon->SetFillStyle(3001);
    h_dt_low_muon->GetXaxis()->SetTitleSize(0.04);
    h_dt_low_muon->GetYaxis()->SetTitleSize(0.05);
    h_dt_low_muon->GetXaxis()->SetLabelSize(0.04);
    h_dt_low_muon->GetYaxis()->SetLabelSize(0.04);
    h_dt_low_muon->SetTitle("#Delta t: Low Energy Isolated to Muon");

    h_dt_low_muon->Draw("HIST");

    TF1* expFit_low_muon = nullptr;
    if (h_dt_low_muon->GetEntries() > 10) {
        expFit_low_muon = new TF1("expFit_low_muon", ExpFit, FIT_MIN_LOW_MUON, FIT_MAX_LOW_MUON, 3);
        expFit_low_muon->SetParNames("N_{0}", "#tau", "C");

        double integral = h_dt_low_muon->Integral(h_dt_low_muon->FindBin(FIT_MIN_LOW_MUON), h_dt_low_muon->FindBin(FIT_MAX_LOW_MUON));
        double bin_width = h_dt_low_muon->GetBinWidth(1);
        double C_init = 0;
        int bin_400 = h_dt_low_muon->FindBin(400.0);
        int bin_500 = h_dt_low_muon->FindBin(500.0);
        double min_content = 1e9;
        for (int i = bin_400; i <= bin_500; i++) {
            double content = h_dt_low_muon->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1;

        double N0_init = (integral * bin_width - C_init * (FIT_MAX_LOW_MUON - FIT_MIN_LOW_MUON)) / 200.0;
        if (N0_init <= 0) N0_init = 1.0;

        expFit_low_muon->SetParameters(N0_init, 200.0, C_init);
        expFit_low_muon->SetParLimits(0, 0, N0_init * 100);
        expFit_low_muon->SetParLimits(1, 0.1, 1000.0);
        expFit_low_muon->SetParLimits(2, -C_init * 10, C_init * 10);

        int fitStatus = h_dt_low_muon->Fit(expFit_low_muon, "RE+", "", FIT_MIN_LOW_MUON, FIT_MAX_LOW_MUON);

        expFit_low_muon->SetLineColor(kRed);
        expFit_low_muon->SetLineWidth(3);
        expFit_low_muon->Draw("SAME");

        gPad->Update();
        TPaveStats *stats = (TPaveStats*)h_dt_low_muon->FindObject("stats");
        if (stats) {
            stats->SetX1NDC(0.6);
            stats->SetX2NDC(0.9);
            stats->SetY1NDC(0.7);
            stats->SetY2NDC(0.95);
            stats->SetTextColor(kRed);
        }

        double N0 = expFit_low_muon->GetParameter(0);
        double N0_err = expFit_low_muon->GetParError(0);
        double tau = expFit_low_muon->GetParameter(1);
        double tau_err = expFit_low_muon->GetParError(1);
        double C = expFit_low_muon->GetParameter(2);
        double C_err = expFit_low_muon->GetParError(2);
        double chi2 = expFit_low_muon->GetChisquare();
        int ndf = expFit_low_muon->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Exponential Fit Results (Low to Muon dt, " << FIT_MIN_LOW_MUON << "-" << FIT_MAX_LOW_MUON << " µs):\n";
        cout << "Fit Status: " << fitStatus << " (0 = success)\n";
        cout << Form("N_{0} = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;

        // ==== NEUTRON PURITY ANALYSIS ====
        cout << "=== Neutron Purity Analysis ===" << endl;

        double bw = h_dt_low_muon->GetBinWidth(1);
        double N0_rate = N0;
        double C_rate = C;
        double t_min = 16.0;

        for (int time_cut = 16; time_cut <= 1000; time_cut += 10) {
            double sig = N0_rate * exp(-time_cut / tau);
            double bkg = C_rate;
            
            if (bkg > 0 && sig > 0) {
                double neutron_ratio = sig / bkg;
                double significance = sig / sqrt(sig + bkg);
                h_neutron_richness->Fill(time_cut, neutron_ratio);
                h_signal_significance->Fill(time_cut, significance);
            }
            
            if (time_cut % 100 == 0) {
                cout << "Time cut " << time_cut << " µs: Signal=" << sig 
                     << ", Bkg=" << bkg << ", Ratio=" << sig/bkg 
                     << ", Significance=" << sig/sqrt(sig + bkg) << endl;
            }
        }
    } else {
        cout << "Warning: h_dt_low_muon has insufficient entries (" << h_dt_low_muon->GetEntries() 
             << "), skipping exponential fit" << endl;
    }

    c_low_muon->Update();
    plotName = OUTPUT_DIR + "/DeltaT_Low_to_Muon.png";
    c_low_muon->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    if (expFit_low_muon) delete expFit_low_muon;

    TCanvas *c_high_muon = new TCanvas("c_high_muon", "DeltaT High to Muon", 1200, 800);
    c_high_muon->SetLeftMargin(0.12);
    c_high_muon->SetRightMargin(0.08);
    c_high_muon->SetBottomMargin(0.12);
    c_high_muon->SetTopMargin(0.08);

    h_dt_high_muon->SetLineWidth(2);
    h_dt_high_muon->SetLineColor(kBlue);
    h_dt_high_muon->SetFillColor(kBlack);
    h_dt_high_muon->SetFillStyle(3001);
    h_dt_high_muon->GetXaxis()->SetTitleSize(0.05);
    h_dt_high_muon->GetYaxis()->SetTitleSize(0.05);
    h_dt_high_muon->GetXaxis()->SetLabelSize(0.04);
    h_dt_high_muon->GetYaxis()->SetLabelSize(0.04);
    h_dt_high_muon->SetTitle("#Delta t: High Energy Isolated to Muon");

    h_dt_high_muon->Draw("HIST");

    TF1* expFit_high_muon = nullptr;
    if (h_dt_high_muon->GetEntries() > 10) {
        expFit_high_muon = new TF1("expFit_high_muon", ExpFit, FIT_MIN_LOW_MUON, FIT_MAX_LOW_MUON, 3);
        expFit_high_muon->SetParNames("N_{0}", "#tau", "C");

        double integral = h_dt_high_muon->Integral(h_dt_high_muon->FindBin(FIT_MIN_LOW_MUON), h_dt_high_muon->FindBin(FIT_MAX_LOW_MUON));
        double bin_width = h_dt_high_muon->GetBinWidth(1);
        double C_init = 0;
        int bin_400 = h_dt_high_muon->FindBin(400.0);
        int bin_500 = h_dt_high_muon->FindBin(500.0);
        double min_content = 1e9;
        for (int i = bin_400; i <= bin_500; i++) {
            double content = h_dt_high_muon->GetBinContent(i);
            if (content > 0 && content < min_content) min_content = content;
        }
        if (min_content < 1e9) C_init = min_content;
        else C_init = 0.1;

        double N0_init = (integral * bin_width - C_init * (FIT_MAX_LOW_MUON - FIT_MIN_LOW_MUON)) / 200.0;
        if (N0_init <= 0) N0_init = 1.0;

        expFit_high_muon->SetParameters(N0_init, 200.0, C_init);
        expFit_high_muon->SetParLimits(0, 0, N0_init * 100);
        expFit_high_muon->SetParLimits(1, 0.1, 1000.0);
        expFit_high_muon->SetParLimits(2, -C_init * 10, C_init * 10);

        int fitStatus = h_dt_high_muon->Fit(expFit_high_muon, "RE+", "", FIT_MIN_LOW_MUON, FIT_MAX_LOW_MUON);

        expFit_high_muon->SetLineColor(kRed);
        expFit_high_muon->SetLineWidth(3);
        expFit_high_muon->Draw("SAME");

        gPad->Update();
        TPaveStats *stats = (TPaveStats*)h_dt_high_muon->FindObject("stats");
        if (stats) {
            stats->SetX1NDC(0.6);
            stats->SetX2NDC(0.9);
            stats->SetY1NDC(0.7);
            stats->SetY2NDC(0.95);
            stats->SetTextColor(kRed);
        }

        double N0 = expFit_high_muon->GetParameter(0);
        double N0_err = expFit_high_muon->GetParError(0);
        double tau = expFit_high_muon->GetParameter(1);
        double tau_err = expFit_high_muon->GetParError(1);
        double C = expFit_high_muon->GetParameter(2);
        double C_err = expFit_high_muon->GetParError(2);
        double chi2 = expFit_high_muon->GetChisquare();
        int ndf = expFit_high_muon->GetNDF();
        double chi2_ndf = ndf > 0 ? chi2 / ndf : 0;

        cout << "Exponential Fit Results (High to Muon dt, " << FIT_MIN_LOW_MUON << "-" << FIT_MAX_LOW_MUON << " µs):\n";
        cout << "Fit Status: " << fitStatus << " (0 = success)\n";
        cout << Form("N_{0} = %.1f ± %.1f", N0, N0_err) << endl;
        cout << Form("τ = %.4f ± %.4f µs", tau, tau_err) << endl;
        cout << Form("C = %.1f ± %.1f", C, C_err) << endl;
        cout << Form("χ²/NDF = %.4f", chi2_ndf) << endl;
        cout << "----------------------------------------" << endl;
    } else {
        cout << "Warning: h_dt_high_muon has insufficient entries (" << h_dt_high_muon->GetEntries() 
             << "), skipping exponential fit" << endl;
    }

    c_high_muon->Update();
    plotName = OUTPUT_DIR + "/DeltaT_High_to_Muon.png";
    c_high_muon->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    if (expFit_high_muon) delete expFit_high_muon;

    // ==== NEW MICHEL BACKGROUND SUBTRACTION FOR LOW ENERGY SIDEBAND ====
    cout << "=== Michel Background Subtraction for Low Energy Sideband Analysis ===" << endl;

    double N0_fit = 0, tau_fit = 0, C_fit = 0;
    double N0_err = 0, tau_err = 0, C_err = 0;
    double chi2_ndf_fit = 0;
    double predicted_michels = 0;

    if (h_dt_michel_sideband->GetEntries() > 20) {
        TF1* michel_fit = new TF1("michel_fit", ExpFit, 0.76, 16.0, 3);
        michel_fit->SetParNames("N_{0}", "#tau", "C");
        
        // Better initial parameters for Michel decay
        double integral = h_dt_michel_sideband->Integral();
        double bin_width = h_dt_michel_sideband->GetBinWidth(1);
        double N0_init = integral * 2.2; // Rough estimate
        double tau_init = 2.2; // Michel lifetime
        double C_init = h_dt_michel_sideband->GetBinContent(h_dt_michel_sideband->FindBin(15.0)); // Late time background
        
        michel_fit->SetParameters(N0_init, tau_init, C_init);
        michel_fit->SetParLimits(0, 0, N0_init * 10);
        michel_fit->SetParLimits(1, 1.0, 3.0); // Constrain to physical Michel lifetime
        michel_fit->SetParLimits(2, 0, C_init * 5);
        
        int fit_status = h_dt_michel_sideband->Fit(michel_fit, "RE+", "", 0.76, 16.0);
        
        if (fit_status == 0) {
            N0_fit = michel_fit->GetParameter(0);
            tau_fit = michel_fit->GetParameter(1);
            C_fit = michel_fit->GetParameter(2);
            N0_err = michel_fit->GetParError(0);
            tau_err = michel_fit->GetParError(1);
            C_err = michel_fit->GetParError(2);
            chi2_ndf_fit = michel_fit->GetChisquare() / michel_fit->GetNDF();
            
            cout << "Michel Fit Results (0-16 μs sideband):" << endl;
            cout << Form("N₀ = %.1f ± %.1f", N0_fit, N0_err) << endl;
            cout << Form("τ = %.3f ± %.3f μs", tau_fit, tau_err) << endl;
            cout << Form("C = %.1f ± %.1f", C_fit, C_err) << endl;
            cout << Form("χ²/NDF = %.2f", chi2_ndf_fit) << endl;
            
            // Predict Michel events in signal region (16-100 μs)
            if (tau_fit > 0) {
                predicted_michels = N0_fit * (exp(-SIGNAL_REGION_MIN/tau_fit) - exp(-SIGNAL_REGION_MAX/tau_fit));
                //predicted_michels += C_fit * (SIGNAL_REGION_MAX - SIGNAL_REGION_MIN); // Constant background
            }
            
            cout << Form("Predicted Michel events in 16-100 μs: %.1f", predicted_michels) << endl;
            
        } else {
            cout << "Warning: Michel fit failed with status " << fit_status << endl;
        }
        
        delete michel_fit;
    } else {
        cout << "Warning: Insufficient Michel events in sideband for fitting: " 
             << h_dt_michel_sideband->GetEntries() << endl;
    }

    // ==== PERFORM FINAL BACKGROUND SUBTRACTION FOR LOW ENERGY ANALYSIS ====
    double signal_events = calculateTotalEvents(h_low_pe_signal);
    double sideband_events = calculateTotalEvents(h_low_pe_sideband);
    
    // Scale neutron-free background
    TH1D* h_scaled_sideband = (TH1D*)h_low_pe_sideband->Clone("scaled_sideband");
    double neutron_free_scale = (SIGNAL_REGION_MAX - SIGNAL_REGION_MIN) / (1200.0 - 1000.0); // 84/200 = 0.42
    h_scaled_sideband->Scale(neutron_free_scale);
    
    // Create Michel background template for 16-100 μs prediction
    TH1D* h_michel_background_predicted = (TH1D*)h_michel_energy_sideband->Clone("michel_background_predicted");
    TH1D* h_michel_energy_predicted = (TH1D*)h_michel_energy_sideband->Clone("michel_energy_predicted");

    double sideband_integral = h_dt_michel_sideband->Integral();
    double michel_scale = (predicted_michels > 0 && sideband_integral > 0) ? predicted_michels / sideband_integral : 0;
    h_michel_background_predicted->Scale(michel_scale);
    h_michel_energy_predicted->Scale(michel_scale);
    
    // Final subtraction: Signal - NeutronFree - MichelBackground
    TH1D* h_final_subtracted = (TH1D*)h_low_pe_signal->Clone("final_subtracted");
    h_final_subtracted->Add(h_scaled_sideband, -1);
    h_final_subtracted->Add(h_michel_background_predicted, -1);
    
    double final_events = calculateTotalEvents(h_final_subtracted);
    
    cout << "=== Final Low Energy Subtraction Results ===" << endl;
    cout << "Signal region (16-100 μs) events: " << signal_events << endl;
    cout << "Scaled neutron-free background: " << calculateTotalEvents(h_scaled_sideband) << endl;
    cout << "Predicted Michel background: " << predicted_michels << endl;
    cout << "Michel background scaling factor: " << michel_scale << endl;
    cout << "Final subtracted events: " << final_events << endl;

    // ==== PLOT THE NEW MICHEL BACKGROUND SUBTRACTION ====
    TCanvas *c_michel_method = new TCanvas("c_michel_method", "Michel Background Subtraction Method", 1200, 800);
    c_michel_method->Divide(2, 1);
    
    // Plot 1: Time distribution fit
    c_michel_method->cd(1);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.08);
    gPad->SetBottomMargin(0.12);
    gPad->SetTopMargin(0.08);
    
    h_dt_michel_sideband->SetLineColor(kBlue);
    h_dt_michel_sideband->SetLineWidth(2);
    h_dt_michel_sideband->Draw("HIST");
    
    // Redraw fit if successful
    if (predicted_michels > 0) {
        TF1* michel_fit_plot = new TF1("michel_fit_plot", ExpFit, 0.76, 16.0, 3);
        michel_fit_plot->SetParameters(N0_fit, tau_fit, C_fit);
        michel_fit_plot->SetLineColor(kRed);
        michel_fit_plot->SetLineWidth(2);
        michel_fit_plot->Draw("SAME");
    }
    
    // Plot 2: Energy spectrum comparison (0-16 μs actual vs 16-100 μs predicted)
    c_michel_method->cd(2);
    gPad->SetLeftMargin(0.12);
    gPad->SetRightMargin(0.08);
    gPad->SetBottomMargin(0.12);
    gPad->SetTopMargin(0.08);
    
    h_michel_energy_sideband->SetLineColor(kBlue);
    h_michel_energy_sideband->SetLineWidth(2);
    h_michel_energy_sideband->SetStats(0);
    h_michel_energy_sideband->Draw("HIST");

    // Plot predicted Michel spectrum for 16-100 μs
    h_michel_energy_predicted->SetLineColor(kRed);
    h_michel_energy_predicted->SetLineWidth(2);
    h_michel_energy_predicted->SetLineStyle(2); // Dashed for prediction
    h_michel_energy_predicted->SetStats(0);
    h_michel_energy_predicted->Draw("HIST SAME");
    
    // Add clean legend
    TLegend *leg_energy = new TLegend(0.15, 0.80, 0.45, 0.93);
    leg_energy->SetBorderSize(0);
    leg_energy->SetFillStyle(0);
    leg_energy->SetTextSize(0.045);
    leg_energy->AddEntry(h_michel_energy_sideband, "Michel (0-16 #mus) Actual", "l");
    leg_energy->AddEntry(h_michel_energy_predicted, "Michel (16-100 #mus) Predicted", "l");
    leg_energy->Draw();
    
    c_michel_method->Update();
    plotName = OUTPUT_DIR + "/Michel_Background_Subtraction.png";
    c_michel_method->SaveAs(plotName.c_str());
    cout << "Saved Michel background subtraction plot: " << plotName << endl;

    // ==== RETAIN ALL ORIGINAL LOW ENERGY PLOTS ====
    
    c->Clear();
    h_isolated_ge40->SetLineColor(kBlack);
    h_isolated_ge40->Draw();
    c->Update();
    plotName = OUTPUT_DIR + "/Isolated_GE40_PE.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Plot Neutron Purity Analysis with larger axis titles
    c->Clear();
    c->Divide(1,2);
    
    // First plot: Neutron Richness
    c->cd(1);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    h_neutron_richness->SetStats(0);
    h_neutron_richness->SetLineColor(kBlue);
    h_neutron_richness->SetLineWidth(3);
    h_neutron_richness->GetXaxis()->SetTitleSize(0.08);
    h_neutron_richness->GetXaxis()->SetTitleOffset(0.6);
    h_neutron_richness->GetYaxis()->SetTitleSize(0.08);
    h_neutron_richness->GetYaxis()->SetTitleOffset(0.6);
    h_neutron_richness->GetXaxis()->SetLabelSize(0.05);
    h_neutron_richness->GetYaxis()->SetLabelSize(0.05);
    h_neutron_richness->Draw("HIST");
    
    // Second plot: Signal Significance
    c->cd(2);
    gPad->SetLeftMargin(0.10);
    gPad->SetBottomMargin(0.10);
    h_signal_significance->SetStats(0);
    h_signal_significance->SetLineColor(kRed);
    h_signal_significance->SetLineWidth(3);
    h_signal_significance->GetXaxis()->SetTitleSize(0.08);
    h_signal_significance->GetXaxis()->SetTitleOffset(0.6);
    h_signal_significance->GetYaxis()->SetTitleSize(0.08);
    h_signal_significance->GetYaxis()->SetTitleOffset(0.6);
    h_signal_significance->GetXaxis()->SetLabelSize(0.05);
    h_signal_significance->GetYaxis()->SetLabelSize(0.05);
    h_signal_significance->Draw("HIST");
    
    c->Update();
    plotName = OUTPUT_DIR + "/Neutron_Purity_Analysis.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Plot Multi-dimensional Cuts
    c->Clear();
    c->Divide(1,2);
    c->cd(1);
    h_energy_vs_time_low->Draw("COLZ");
    c->cd(2);
    h_energy_vs_time_high->Draw("COLZ");
    c->Update();
    plotName = OUTPUT_DIR + "/Energy_vs_Time_MultiD.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // ==== ORIGINAL LOW ENERGY SIDEBAND SUBTRACTION PLOTS (RETAINED) ====
    
    // Plot 1: Original sideband subtraction
    TCanvas *c_sideband1 = new TCanvas("c_sideband1", "Low Energy Sideband Subtraction", 1200, 800);
    c_sideband1->SetLeftMargin(0.1);
    c_sideband1->SetRightMargin(0.1);
    c_sideband1->SetBottomMargin(0.1);
    c_sideband1->SetTopMargin(0.1);

    h_low_pe_signal->SetLineColor(kRed);
    h_low_pe_signal->SetLineWidth(3);
    h_low_pe_sideband->SetLineColor(kBlue);
    h_low_pe_sideband->SetLineWidth(3);
    h_scaled_sideband->SetLineColor(kBlue);
    h_scaled_sideband->SetLineWidth(3);
    h_scaled_sideband->SetLineStyle(2);

    h_low_pe_signal->SetStats(0);
    h_low_pe_sideband->SetStats(0);
    h_scaled_sideband->SetStats(0);

    h_low_pe_signal->Draw("HIST");
    h_low_pe_sideband->Draw("HIST same");
    h_scaled_sideband->Draw("HIST same");

    TLegend *leg_sub1 = new TLegend(0.5, 0.65, 0.9, 0.9);
    leg_sub1->SetTextSize(0.025);
    leg_sub1->SetTextFont(42);
    leg_sub1->SetBorderSize(1);
    leg_sub1->SetFillStyle(0);
    leg_sub1->AddEntry(h_low_pe_signal, Form("Neutron rich region (16-100 #mus) [%.0f events]", signal_events), "l");
    leg_sub1->AddEntry(h_low_pe_sideband, Form("Neutron free region (1000-1200 #mus) [%.0f events]", sideband_events), "l");
    leg_sub1->AddEntry(h_scaled_sideband, Form("Scaled neutron free region [%.1f events]", calculateTotalEvents(h_scaled_sideband)), "l");
    leg_sub1->Draw();

    c_sideband1->Update();
    plotName = OUTPUT_DIR + "/Low_Energy_Sideband_Subtraction.png";
    c_sideband1->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Plot 2: Complete three-component subtraction (original method)
    TCanvas *c_sideband2 = new TCanvas("c_sideband2", "Low Energy Sideband Subtraction with Michel Background", 1200, 800);
    c_sideband2->SetLeftMargin(0.1);
    c_sideband2->SetRightMargin(0.1);
    c_sideband2->SetBottomMargin(0.1);
    c_sideband2->SetTopMargin(0.1);

    h_low_pe_signal->SetLineColor(kRed);
    h_low_pe_signal->SetLineWidth(3);
    h_scaled_sideband->SetLineColor(kBlue);
    h_scaled_sideband->SetLineWidth(2);
    h_scaled_sideband->SetLineStyle(2);
    h_michel_background_predicted->SetLineColor(kMagenta);
    h_michel_background_predicted->SetLineWidth(2);
    h_michel_background_predicted->SetLineStyle(3);
    h_final_subtracted->SetLineColor(kGreen);
    h_final_subtracted->SetLineWidth(3);

    h_low_pe_signal->SetStats(0);
    h_scaled_sideband->SetStats(0);
    h_michel_background_predicted->SetStats(0);
    h_final_subtracted->SetStats(0);

    h_low_pe_signal->Draw("HIST");
    h_scaled_sideband->Draw("HIST SAME");
    h_michel_background_predicted->Draw("HIST SAME");
    h_final_subtracted->Draw("HIST SAME");

    TLegend *leg_sub2 = new TLegend(0.5, 0.6, 0.9, 0.9);
    leg_sub2->SetTextSize(0.025);
    leg_sub2->SetTextFont(42);
    leg_sub2->SetBorderSize(1);
    leg_sub2->SetFillStyle(0);
    leg_sub2->AddEntry(h_low_pe_signal, Form("Neutron rich region (16-100 #mus) [%.0f events]", signal_events), "l");
    leg_sub2->AddEntry(h_scaled_sideband, Form("Scaled neutron free region [%.1f events]", calculateTotalEvents(h_scaled_sideband)), "l");
    leg_sub2->AddEntry(h_michel_background_predicted, Form("Michel background (16-100 #mus) [%.1f events]", predicted_michels), "l");
    leg_sub2->AddEntry(h_final_subtracted, Form("Final: Signal - ScaledBkg - Michel [%.1f events]", final_events), "l");
    leg_sub2->Draw();

    c_sideband2->Update();
    plotName = OUTPUT_DIR + "/Low_Energy_Sideband_Subtraction_Complete.png";
    c_sideband2->SaveAs(plotName.c_str());
    cout << "Saved plot: " << plotName << endl;

    // Cleanup
    delete h_muon_energy;
    delete h_muon_all;
    delete h_michel_energy;
    delete h_dt_michel;
    delete h_energy_vs_dt;
    delete h_side_vp_muon;
    delete h_top_vp_muon;
    delete h_trigger_bits;
    delete h_isolated_pe;
    delete h_low_iso;
    delete h_high_iso;
    delete h_dt_prompt_delayed;
    delete h_dt_low_muon;
    delete h_dt_high_muon;
    delete h_low_pe_signal;
    delete h_low_pe_sideband;
    delete h_isolated_ge40;
    for (int i = 0; i < 10; i++) {
        delete h_veto_panel[i];
    }

    // Cleanup extended Michel histograms
    delete h_dt_michel_sideband;
    delete h_michel_energy_sideband;
    delete h_michel_energy_predicted;
    delete h_final_subtracted;

    // Cleanup new histograms
    delete h_neutron_richness;
    delete h_signal_significance;
    delete h_energy_vs_time_low;
    delete h_energy_vs_time_high;

    delete h_scaled_sideband;
    delete h_michel_background_predicted;
    delete leg_energy;
    delete leg_sub1;
    delete leg_sub2;
    delete c;
    delete c_low_muon;
    delete c_high_muon;
    delete c_michel_method;
    delete c_sideband1;
    delete c_sideband2;

    cout << "Analysis complete. Results saved in " << OUTPUT_DIR << "/ (*.png, *.csv)" << endl;
    return 0;
}
