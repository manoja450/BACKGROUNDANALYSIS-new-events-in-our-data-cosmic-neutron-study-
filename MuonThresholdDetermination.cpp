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
#include <TPad.h>
#include <TFitResult.h>

using std::cout;
using std::endl;
using namespace std;

// Constants for veto panel analysis
const int N_PMTS = 12;
const int PMT_CHANNEL_MAP[12] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};
const int PULSE_THRESHOLD = 30;
const int BS_UNCERTAINTY = 5;
const int EV61_THRESHOLD = 1200;
const double MUON_ENERGY_THRESHOLD = 50;
const int ADCSIZE = 45;

// Generate unique output directory
string getTimestamp() {
    time_t now = time(nullptr);
    struct tm *t = localtime(&now);
    char buffer[20];
    strftime(buffer, sizeof(buffer), "%Y%m%d_%H%M%S", t);
    return string(buffer);
}
const string OUTPUT_DIR = "./MuonThresholdDetermination_" + getTimestamp();

// Veto panel thresholds
const std::vector<double> SIDE_VP_THRESHOLDS = {1100, 1100, 1500, 1800, 850, 900, 900, 800};
const double TOP_VP_THRESHOLD = 600;

// Forward declarations
void createSummaryCanvas(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, TF1* fit_functions[9], const string& outputDir);
TF1* createTopVetoCombinedPlot(TH1D* h_top_veto_combined, const string& outputDir);
void createVetoPanelPlots(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, const string& outputDir);

// Pulse structure
struct pulse {
    double start;
    double end;
    double peak;
    double energy;
    double number;
    bool single;
    bool beam;
    double trigger;
    double side_vp_energy;
    double top_vp_energy;
    double all_vp_energy;
    double last_muon_time;
    bool is_muon;
    bool is_michel;
    bool veto_hit[10];
};

// Temporary pulse structure
struct pulse_temp {
    double start;
    double end;
    double peak;
    double energy;
};

// Landau fit function
Double_t LandauFit(Double_t *x, Double_t *par) {
    // par[0] = normalization
    // par[1] = MPV (Most Probable Value)
    // par[2] = width
    // par[3] = constant background
    
    Double_t landau = TMath::Landau(x[0], par[1], par[2], kTRUE);
    return par[0] * landau + par[3];
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

// New function to create summary canvas with all 9 plots
void createSummaryCanvas(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, TF1* fit_functions[9], const string& outputDir) {
    cout << "\n=== Creating Summary Canvas with All 9 Veto Panel Plots ===" << endl;
    
    // Create a large canvas divided into 3x3 grid
    TCanvas *c_summary = new TCanvas("c_veto_summary", 
                                    "Summary - All Veto Panel Energy Distributions with Landau Fits", 
                                    2400, 1800);
    
    // Divide canvas into 3x3 grid
    c_summary->Divide(3, 3);
    
    // Configure style for summary plots - SAME AS INDIVIDUAL PLOTS
    gStyle->SetOptStat(1110); // Only show entries, mean, std dev (SAME AS INDIVIDUAL)
    gStyle->SetOptFit(0);     // Hide fit statistics completely (SAME AS INDIVIDUAL)
    
    // Plot side veto panels (12-19) in first 8 pads
    for (int i = 0; i < 8; i++) {
        if (h_veto_panel[i]->GetEntries() < 10) {
            cout << "Skipping veto panel " << i+12 << " in summary - insufficient entries" << endl;
            continue;
        }
        
        c_summary->cd(i+1); // Pads 1-8
        
        // Set log scale for current pad
        gPad->SetLogy();
        
        // Configure histogram appearance
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(1);
        h_veto_panel[i]->SetFillColor(kBlue);
        h_veto_panel[i]->SetFillStyle(3003);
        
        // Draw histogram (this will automatically show the default stat box)
        h_veto_panel[i]->Draw("hist");
        
        // Draw fit function if available
        if (fit_functions[i]) {
            fit_functions[i]->SetLineColor(kRed);
            fit_functions[i]->SetLineWidth(2);
            fit_functions[i]->Draw("same");
        }
        
        // Update the pad to ensure stat box is drawn
        gPad->Update();
    }
    
    // Plot combined top veto panels (20+21) in the last pad (position 9)
    c_summary->cd(9);
    gPad->SetLogy();
    
    // Configure combined top veto histogram
    h_top_veto_combined->SetLineColor(kBlack);
    h_top_veto_combined->SetLineWidth(1);
    h_top_veto_combined->SetFillColor(kGreen);
    h_top_veto_combined->SetFillStyle(3003);
    
    // Draw histogram (this will automatically show the default stat box)
    h_top_veto_combined->Draw("hist");
    
    // Use the stored fit function for combined top veto (position 8 in fit_functions array)
    if (fit_functions[8]) {
        fit_functions[8]->SetLineColor(kRed);
        fit_functions[8]->SetLineWidth(2);
        fit_functions[8]->Draw("same");
    }
    
    // Update the pad to ensure stat box is drawn
    gPad->Update();
    
    // Update the canvas
    c_summary->Update();
    
    // Save the summary canvas
    string summaryName = outputDir + "/All_Veto_Panels_Summary.png";
    c_summary->SaveAs(summaryName.c_str());
    cout << "Saved summary canvas: " << summaryName << endl;
    
    // Also save as PDF
    string summaryPdf = outputDir + "/All_Veto_Panels_Summary.pdf";
    c_summary->SaveAs(summaryPdf.c_str());
    
    delete c_summary;
    cout << "=== Summary Canvas Complete ===" << endl;
}

// Modified to return the fit function
TF1* createTopVetoCombinedPlot(TH1D* h_top_veto_combined, const string& outputDir) {
    cout << "\n=== Creating Combined Top Veto Plot ===" << endl;
    cout << "Entries in combined histogram: " << h_top_veto_combined->GetEntries() << endl;
    
    if (h_top_veto_combined->GetEntries() < 10) {
        cout << "WARNING: Combined top veto plot has very few entries: " 
             << h_top_veto_combined->GetEntries() << endl;
        return nullptr;
    }
    
    TCanvas *c = new TCanvas("c_top_veto_combined", 
                            "Combined Top Veto Panels 20+21 - Muon Energy Deposition", 
                            1200, 800);
    
    // Configure style - ONLY show basic histogram stats, NO fit stats
    gStyle->SetOptStat(1110); // Only show entries, mean, std dev
    gStyle->SetOptFit(0);     // Hide fit statistics completely
    
    // Set log scale
    c->SetLogy();
    
    // Configure histogram
    h_top_veto_combined->SetLineColor(kBlack);
    h_top_veto_combined->SetLineWidth(2);
    h_top_veto_combined->SetFillColor(kGreen);
    h_top_veto_combined->SetFillStyle(3003);
    
    // Draw histogram first (ALL DATA)
    h_top_veto_combined->Draw("hist");
    
    // Get histogram properties
    double hist_max = h_top_veto_combined->GetMaximum();
    double x_min = h_top_veto_combined->GetXaxis()->GetXmin();
    double x_max = h_top_veto_combined->GetXaxis()->GetXmax();
    
    // Create a TEMPORARY histogram with only data ABOVE THRESHOLD for fitting
    TH1D* h_top_veto_fit = new TH1D("h_top_veto_fit", "Temporary fit histogram", 
                                   h_top_veto_combined->GetNbinsX(), 
                                   h_top_veto_combined->GetXaxis()->GetXmin(),
                                   h_top_veto_combined->GetXaxis()->GetXmax());
    
    // Copy only bins ABOVE THRESHOLD to the temporary histogram
    int threshold_bin = h_top_veto_combined->FindBin(TOP_VP_THRESHOLD);
    int last_bin = h_top_veto_combined->GetNbinsX();
    
    for (int bin = threshold_bin; bin <= last_bin; bin++) {
        h_top_veto_fit->SetBinContent(bin, h_top_veto_combined->GetBinContent(bin));
        h_top_veto_fit->SetBinError(bin, h_top_veto_combined->GetBinError(bin));
    }
    
    cout << "Original histogram entries: " << h_top_veto_combined->GetEntries() << endl;
    cout << "Fit histogram entries (above threshold): " << h_top_veto_fit->GetEntries() << endl;
    cout << "Threshold: " << TOP_VP_THRESHOLD << " ADC" << endl;
    cout << "Fit range: " << TOP_VP_THRESHOLD << " to " << x_max << " ADC" << endl;
    
    // Get properties from the FIT histogram (above threshold only)
    double fit_hist_max = h_top_veto_fit->GetMaximum();
    double fit_mean = h_top_veto_fit->GetMean();
    double fit_rms = h_top_veto_fit->GetRMS();
    int fit_max_bin = h_top_veto_fit->GetMaximumBin();
    double mpv_guess = h_top_veto_fit->GetBinCenter(fit_max_bin);
    
    // Estimate background from the edges of the FIT histogram
    double background_guess = 0;
    int n_bkg_bins = 5;
    int start_bkg_bin = h_top_veto_fit->FindBin(TOP_VP_THRESHOLD);
    for (int i = start_bkg_bin; i < start_bkg_bin + n_bkg_bins; i++) {
        background_guess += h_top_veto_fit->GetBinContent(i);
    }
    background_guess /= n_bkg_bins;
    
    cout << "Fit histogram info (above threshold only):" << endl;
    cout << "  Max bin content: " << fit_hist_max << endl;
    cout << "  Mean: " << fit_mean << " ADC" << endl;
    cout << "  RMS: " << fit_rms << " ADC" << endl;
    cout << "  MPV guess: " << mpv_guess << " ADC" << endl;
    cout << "  Background guess: " << background_guess << endl;
    
    // Create Landau fit function for fitting
    TF1 *landauFit = new TF1("landauFit_top_combined", LandauFit, x_min, x_max, 4);
    
    // Set initial parameters based on FIT histogram (above threshold only)
    double norm_guess = (fit_hist_max - background_guess) * fit_rms * 2.5;
    double width_guess = fit_rms * 0.8;
    
    landauFit->SetParameters(norm_guess, mpv_guess, width_guess, background_guess);
    
    // Set parameter names
    landauFit->SetParNames("Norm", "MPV", "Width", "Background");
    
    // Set reasonable parameter limits
    landauFit->SetParLimits(0, norm_guess * 0.1, norm_guess * 10);
    landauFit->SetParLimits(1, fit_mean - fit_rms, fit_mean + fit_rms * 2);
    landauFit->SetParLimits(2, width_guess * 0.1, width_guess * 5);
    landauFit->SetParLimits(3, 0, fit_hist_max * 0.5);
    
    landauFit->SetLineColor(kRed);
    landauFit->SetLineWidth(3);
    landauFit->SetNpx(1000);
    
    cout << "Initial fit parameters:" << endl;
    cout << "  Norm: " << norm_guess << endl;
    cout << "  MPV: " << mpv_guess << endl;
    cout << "  Width: " << width_guess << endl;
    cout << "  Background: " << background_guess << endl;
    
    // Perform fit on the TEMPORARY histogram (ABOVE THRESHOLD ONLY)
    cout << "Fitting combined top veto panels (above threshold only)..." << endl;
    Int_t fitStatus = h_top_veto_fit->Fit(landauFit, "SRLN", "", TOP_VP_THRESHOLD, x_max);
    
    // Get fit results
    double mpv = landauFit->GetParameter(1);
    double width = landauFit->GetParameter(2);
    
    // Create a new function for drawing the FULL range
    TF1 *landauDraw = new TF1("landauDraw_top_combined", LandauFit, x_min, x_max, 4);
    
    // Copy parameters from the fit
    for (int i = 0; i < 4; i++) {
        landauDraw->SetParameter(i, landauFit->GetParameter(i));
    }
    
    landauDraw->SetLineColor(kRed);
    landauDraw->SetLineWidth(3);
    landauDraw->SetNpx(1000);
    
    // Draw the FULL fit curve (across entire plot range, but fit was only above threshold)
    landauDraw->Draw("same");
    
    // Update the canvas to ensure everything is drawn
    c->Update();
    
    // Save the plot
    string plotName = outputDir + "/Top_Veto_Panels_20_21_Combined_LandauFit.png";
    c->SaveAs(plotName.c_str());
    cout << "Saved combined top veto plot: " << plotName << endl;
    
    // Also save as PDF
    string pdfName = outputDir + "/Top_Veto_Panels_20_21_Combined_LandauFit.pdf";
    c->SaveAs(pdfName.c_str());
    
    // Print fit results to console only
    if (fitStatus == 0) {
        double mpv_err = landauFit->GetParError(1);
        double chi2 = landauFit->GetChisquare();
        double ndf = landauFit->GetNDF();
        double chi2_ndf = (ndf > 0) ? chi2 / ndf : 0;
        
        cout << "=== Combined Top Veto Panels Fit Results ===" << endl;
        cout << "  MPV = " << mpv << " ± " << mpv_err << " ADC" << endl;
        cout << "  Width = " << width << " ADC" << endl;
        cout << "  Norm = " << landauFit->GetParameter(0) << endl;
        cout << "  Background = " << landauFit->GetParameter(3) << endl;
        cout << "  χ²/NDF = " << chi2_ndf << endl;
        cout << "  Fit Entries = " << h_top_veto_fit->GetEntries() << " (above threshold)" << endl;
        cout << "  Total Entries = " << h_top_veto_combined->GetEntries() << " (all data)" << endl;
        cout << "  Fit Range = " << TOP_VP_THRESHOLD << " - " << x_max << " ADC" << endl;
        cout << "  Threshold = " << TOP_VP_THRESHOLD << " ADC" << endl;
        cout << "=====================================" << endl;
    } else {
        cout << "Fit failed with status: " << fitStatus << endl;
    }
    
    delete landauFit;
    delete h_top_veto_fit;
    delete c;
    
    cout << "=== Combined Top Veto Plot Complete ===" << endl;
    
    // Return the drawing function for use in summary canvas
    return landauDraw;
}

void createVetoPanelPlots(TH1D* h_veto_panel[10], TH1D* h_top_veto_combined, const string& outputDir) {
    cout << "\n=== Creating Veto Panel Plots ===" << endl;
    
    // First, let's check the contents of all veto panels
    cout << "Veto Panel Entries:" << endl;
    for (int i = 0; i < 10; i++) {
        cout << "  Panel " << i+12 << ": " << h_veto_panel[i]->GetEntries() << " entries" << endl;
    }
    cout << "Combined Top: " << h_top_veto_combined->GetEntries() << " entries" << endl;
    
    // Arrays to store fit functions for the summary canvas
    TF1* fit_functions[9] = {nullptr}; // 8 side panels + 1 combined top
    
    // Create individual plots for side veto panels (12-19)
    for (int i = 0; i < 8; i++) {
        if (h_veto_panel[i]->GetEntries() < 10) {
            cout << "Skipping veto panel " << i+12 << " - insufficient entries: " 
                 << h_veto_panel[i]->GetEntries() << endl;
            continue;
        }
        
        cout << "Creating plot for veto panel " << i+12 << "..." << endl;
        
        TCanvas *c = new TCanvas(Form("c_veto_%d", i+12), 
                                Form("Veto Panel %d - Muon Energy Deposition", i+12), 
                                1200, 800);
        
        // Configure style - ONLY show basic histogram stats, NO fit stats
        gStyle->SetOptStat(1110); // Only show entries, mean, std dev
        gStyle->SetOptFit(0);     // Hide fit statistics completely
        
        // Set log scale
        c->SetLogy();
        
        // Configure histogram
        h_veto_panel[i]->SetLineColor(kBlack);
        h_veto_panel[i]->SetLineWidth(2);
        h_veto_panel[i]->SetFillColor(kBlue);
        h_veto_panel[i]->SetFillStyle(3003);
        
        // Draw histogram first (ALL DATA)
        h_veto_panel[i]->Draw("hist");
        
        // Get histogram properties
        double hist_max = h_veto_panel[i]->GetMaximum();
        double x_min = h_veto_panel[i]->GetXaxis()->GetXmin();
        double x_max = h_veto_panel[i]->GetXaxis()->GetXmax();
        
        // Create a TEMPORARY histogram with only data ABOVE THRESHOLD for fitting
        double threshold = SIDE_VP_THRESHOLDS[i];
        TH1D* h_veto_fit = new TH1D(Form("h_veto_fit_%d", i+12), "Temporary fit histogram", 
                                   h_veto_panel[i]->GetNbinsX(), 
                                   h_veto_panel[i]->GetXaxis()->GetXmin(),
                                   h_veto_panel[i]->GetXaxis()->GetXmax());
        
        // Copy only bins ABOVE THRESHOLD to the temporary histogram
        int threshold_bin = h_veto_panel[i]->FindBin(threshold);
        int last_bin = h_veto_panel[i]->GetNbinsX();
        
        for (int bin = threshold_bin; bin <= last_bin; bin++) {
            h_veto_fit->SetBinContent(bin, h_veto_panel[i]->GetBinContent(bin));
            h_veto_fit->SetBinError(bin, h_veto_panel[i]->GetBinError(bin));
        }
        
        // Get properties from the FIT histogram (above threshold only)
        double fit_hist_max = h_veto_fit->GetMaximum();
        double fit_mean = h_veto_fit->GetMean();
        double fit_rms = h_veto_fit->GetRMS();
        int fit_max_bin = h_veto_fit->GetMaximumBin();
        double mpv_guess = h_veto_fit->GetBinCenter(fit_max_bin);
        
        // Estimate background
        double background_guess = 0;
        int n_bkg_bins = 5;
        int start_bkg_bin = h_veto_fit->FindBin(threshold);
        for (int j = start_bkg_bin; j < start_bkg_bin + n_bkg_bins; j++) {
            background_guess += h_veto_fit->GetBinContent(j);
        }
        background_guess /= n_bkg_bins;
        
        // Create Landau fit function for fitting
        TF1 *landauFit = new TF1(Form("landauFit_%d", i+12), LandauFit, x_min, x_max, 4);
        
        // Set improved initial parameters based on FIT histogram
        double norm_guess = (fit_hist_max - background_guess) * fit_rms * 2.5;
        double width_guess = fit_rms * 0.8;
        
        landauFit->SetParameters(norm_guess, mpv_guess, width_guess, background_guess);
        landauFit->SetParNames("Norm", "MPV", "Width", "Background");
        landauFit->SetParLimits(0, norm_guess * 0.1, norm_guess * 10);
        landauFit->SetParLimits(1, fit_mean - fit_rms, fit_mean + fit_rms * 2);
        landauFit->SetParLimits(2, width_guess * 0.1, width_guess * 5);
        landauFit->SetParLimits(3, 0, fit_hist_max * 0.5);
        
        landauFit->SetLineColor(kRed);
        landauFit->SetLineWidth(3);
        landauFit->SetNpx(1000);
        
        // Perform fit to get parameters (without drawing stats, ABOVE THRESHOLD ONLY)
        Int_t fitStatus = h_veto_fit->Fit(landauFit, "SRLN", "", threshold, x_max);
        
        // Create drawing function with full range
        TF1 *landauDraw = new TF1(Form("landauDraw_%d", i+12), LandauFit, x_min, x_max, 4);
        for (int j = 0; j < 4; j++) {
            landauDraw->SetParameter(j, landauFit->GetParameter(j));
        }
        landauDraw->SetLineColor(kRed);
        landauDraw->SetLineWidth(3);
        landauDraw->SetNpx(1000);
        
        // Store the fit function for the summary canvas
        fit_functions[i] = new TF1(*landauDraw);
        
        // Draw the FULL fit curve (across entire plot range, but fit was only above threshold)
        landauDraw->Draw("same");
        
        // Update the canvas
        c->Update();
        
        // Save the plot
        string plotName = outputDir + Form("/Veto_Panel_%d_LandauFit.png", i+12);
        c->SaveAs(plotName.c_str());
        cout << "Saved veto panel plot: " << plotName << endl;
        
        // Print fit results to console only
        if (fitStatus == 0) {
            double mpv = landauFit->GetParameter(1);
            double mpv_err = landauFit->GetParError(1);
            cout << "Veto Panel " << i+12 << " - MPV: " << mpv << " ± " << mpv_err 
                 << ", Fit Range: " << threshold << " - " << x_max << " ADC" << endl;
        }
        
        delete landauFit;
        delete landauDraw;
        delete h_veto_fit;
        delete c;
    }

    // Create combined top veto plot and store its fit function
    cout << "\nCreating combined top veto plot..." << endl;
    fit_functions[8] = createTopVetoCombinedPlot(h_top_veto_combined, outputDir);
    
    // Create summary canvas with all 9 plots (8 side + 1 combined top)
    cout << "\nCreating summary canvas with all 9 veto panel plots..." << endl;
    createSummaryCanvas(h_veto_panel, h_top_veto_combined, fit_functions, outputDir);
    
    // Clean up fit functions
    for (int i = 0; i < 9; i++) {
        if (fit_functions[i]) delete fit_functions[i];
    }
}

int main(int argc, char *argv[]) {
    // Parse command-line arguments
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <input_file1> [<input_file2> ...]" << endl;
        return -1;
    }

    vector<string> inputFiles;
    for (int i = 1; i < argc; i++) {
        inputFiles.push_back(argv[i]);
    }

    // Create output directory
    createOutputDirectory(OUTPUT_DIR);

    cout << "Veto Panel Muon Analysis" << endl;
    cout << "Output directory: " << OUTPUT_DIR << endl;
    cout << "Input files:" << endl;
    for (const auto& file : inputFiles) {
        cout << "  " << file << endl;
    }

    // Statistics counters
    int num_muons = 0;
    int num_events = 0;
    int num_top_veto_hits = 0;

    // Histograms for veto panels (12-21) - WILL STORE ALL DATA
    TH1D* h_veto_panel[10];
    const char* veto_names[10] = {
        "Veto Panel 12 - Energy", "Veto Panel 13 - Energy", 
        "Veto Panel 14 - Energy", "Veto Panel 15 - Energy",
        "Veto Panel 16 - Energy", "Veto Panel 17 - Energy", 
        "Veto Panel 18 - Energy", "Veto Panel 19 - Energy",
        "Veto Panel 20 - Energy", "Veto Panel 21 - Energy"
    };
    
    // Combined histogram for top veto panels (20+21) - WILL STORE ALL DATA
    TH1D* h_top_veto_combined = new TH1D("h_top_veto_combined", 
                                        "Combined Top Veto Panels 20+21 - Energy Deposition;Energy (ADC);Counts", 
                                        200, 200, 3000);
    
    // Initialize veto panel histograms
    for (int i = 0; i < 10; i++) {
        h_veto_panel[i] = new TH1D(Form("h_veto_panel_%d", i+12), 
                                  Form("%s;Energy (ADC);Counts", veto_names[i]), 
                                  200, 200, 5000);
    }

    for (const auto& inputFileName : inputFiles) {
        if (gSystem->AccessPathName(inputFileName.c_str())) {
            cout << "Could not open file: " << inputFileName << ". Skipping..." << endl;
            continue;
        }

        TFile *f = new TFile(inputFileName.c_str());
        cout << "Processing file: " << inputFileName << endl;

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

        for (int iEnt = 0; iEnt < numEntries; iEnt++) {
            t->GetEntry(iEnt);
            num_events++;

            std::vector<double> veto_energies(10, 0);
            TH1D h_wf("h_wf", "Waveform", ADCSIZE, 0, ADCSIZE);

            // Process ALL events and store veto panel energies
            for (int iChan = 0; iChan < 23; iChan++) {
                // Fill waveform histogram
                for (int i = 0; i < ADCSIZE; i++) {
                    h_wf.SetBinContent(i + 1, adcVal[iChan][i] - baselineMean[iChan]);
                }

                // Calculate total energy in waveform
                double allPulseEnergy = 0;
                for (int iBin = 1; iBin <= ADCSIZE; iBin++) {
                    allPulseEnergy += h_wf.GetBinContent(iBin);
                }

                // Store energy for veto panels (ADC) - FOR ALL EVENTS
                if (iChan >= 12 && iChan <= 19) {
                    veto_energies[iChan - 12] = allPulseEnergy;
                } else if (iChan >= 20 && iChan <= 21) {
                    double factor = (iChan == 20) ? 1.07809 : 1.0;
                    veto_energies[iChan - 12] = allPulseEnergy * factor;
                }

                h_wf.Reset();
            }

            // Calculate PMT energy for muon identification
            double pmt_energy = 0;
            for (int iChan = 0; iChan <= 11; iChan++) {
                double allPulseEnergy = 0;
                for (int i = 0; i < ADCSIZE; i++) {
                    allPulseEnergy += (adcVal[iChan][i] - baselineMean[iChan]);
                }
                pmt_energy += allPulseEnergy;
            }

            // FILL ALL VETO PANEL HISTOGRAMS WITH ALL DATA (regardless of muon condition)
            for (int i = 0; i < 10; i++) {
                h_veto_panel[i]->Fill(veto_energies[i]);
            }
            
            // FILL COMBINED TOP VETO HISTOGRAM WITH ALL DATA
            double max_top_energy = std::max(veto_energies[8], veto_energies[9]);
            h_top_veto_combined->Fill(max_top_energy);

            // Muon detection using veto panels (for counting only)
            bool veto_hit = false;
            for (size_t i = 0; i < SIDE_VP_THRESHOLDS.size(); i++) {
                if (veto_energies[i] > SIDE_VP_THRESHOLDS[i]) {
                    veto_hit = true;
                    break;
                }
            }
            if (!veto_hit && (veto_energies[8] > TOP_VP_THRESHOLD || veto_energies[9] > TOP_VP_THRESHOLD)) {
                veto_hit = true;
            }

            if (pmt_energy > MUON_ENERGY_THRESHOLD && veto_hit) {
                num_muons++;
                if (veto_energies[8] > TOP_VP_THRESHOLD || veto_energies[9] > TOP_VP_THRESHOLD) {
                    num_top_veto_hits++;
                }
            }
        }

        cout << "File " << inputFileName << " - Events: " << num_events << ", Muons: " << num_muons 
             << ", Top Veto Hits: " << num_top_veto_hits << endl;
        f->Close();
    }

    // Print detailed statistics before creating plots
    cout << "\n=== Final Statistics Before Plotting ===" << endl;
    cout << "Total events processed: " << num_events << endl;
    cout << "Total muons identified: " << num_muons << endl;
    cout << "Total top veto hits: " << num_top_veto_hits << endl;
    cout << "Combined top veto histogram entries: " << h_top_veto_combined->GetEntries() << endl;
    
    for (int i = 0; i < 10; i++) {
        cout << "Veto Panel " << i+12 << ": " << h_veto_panel[i]->GetEntries() << " entries" << endl;
    }
    cout << "=====================================" << endl;

    // Create plots
    createVetoPanelPlots(h_veto_panel, h_top_veto_combined, OUTPUT_DIR);

    // Print final summary
    cout << "\n=== Analysis Complete ===" << endl;
    cout << "Total muon events: " << num_muons << endl;
    cout << "Top veto hits: " << num_top_veto_hits << endl;
    cout << "Combined top veto entries: " << h_top_veto_combined->GetEntries() << endl;
    cout << "Results saved in: " << OUTPUT_DIR << endl;

    // Clean up
    for (int i = 0; i < 10; i++) delete h_veto_panel[i];
    delete h_top_veto_combined;
    
    return 0;
}
