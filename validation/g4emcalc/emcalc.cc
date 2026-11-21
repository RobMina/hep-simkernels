// Geant4's own muon energy-loss values for the hep-simkernels comparison with the PDG tables: G4EmCalculator
// ComputeDEDX (unrestricted: cut = DBL_MAX) of muIoni, muPairProd and muBrems for mu- and mu+ in NIST materials, at
// the kinetic energies of a file (MeV, one per line), after a zero-event run has built the physics tables.
//   emcalc PHYSICS_LIST ENERGIES_FILE OUT.json MATERIAL [MATERIAL ...]
// Output: {"geant4": version, "physics_list": ..., "rows": [[material, particle, process, T_MeV, MeV cm2/g], ...]}
#include <cfloat>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "G4Box.hh"
#include "G4EmCalculator.hh"
#include "G4LogicalVolume.hh"
#include "G4NistManager.hh"
#include "G4PVPlacement.hh"
#include "G4ParticleGun.hh"
#include "G4ParticleTable.hh"
#include "G4PhysListFactory.hh"
#include "G4RunManager.hh"
#include "G4SystemOfUnits.hh"
#include "G4VUserDetectorConstruction.hh"
#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4Version.hh"

namespace {
std::vector<std::string> gMaterials;

class Detector : public G4VUserDetectorConstruction {
 public:
  G4VPhysicalVolume* Construct() override {
    auto* nist = G4NistManager::Instance();
    auto* world = new G4LogicalVolume(new G4Box("world", 1 * m, 1 * m, 1 * m), nist->FindOrBuildMaterial("G4_Galactic"),
                                      "world");
    auto* pw = new G4PVPlacement(nullptr, {}, world, "world", nullptr, false, 0);
    double x = -0.9 * m;
    for (std::size_t i = 0; i < gMaterials.size(); ++i, x += 0.1 * m) {
      auto* mat = nist->FindOrBuildMaterial(gMaterials[i]);
      if (!mat) throw std::runtime_error("unknown NIST material " + gMaterials[i]);
      auto* lv = new G4LogicalVolume(new G4Box("b" + std::to_string(i), 4 * cm, 4 * cm, 4 * cm), mat, gMaterials[i]);
      new G4PVPlacement(nullptr, {x, 0, 0}, lv, gMaterials[i], world, false, static_cast<int>(i));
    }
    return pw;
  }
};

class Gun : public G4VUserPrimaryGeneratorAction {
 public:
  Gun() : fGun(1) { fGun.SetParticleDefinition(G4ParticleTable::GetParticleTable()->FindParticle("mu-")); }
  void GeneratePrimaries(G4Event* e) override { fGun.GeneratePrimaryVertex(e); }

 private:
  G4ParticleGun fGun;
};
}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: emcalc PHYSICS_LIST ENERGIES_FILE OUT.json MATERIAL [MATERIAL ...]\n";
    return 2;
  }
  for (int i = 4; i < argc; ++i) gMaterials.emplace_back(argv[i]);
  std::vector<double> energies;
  std::ifstream ef(argv[2]);
  for (double t; ef >> t;) energies.push_back(t);
  auto* rm = new G4RunManager;
  rm->SetUserInitialization(new Detector);
  G4PhysListFactory factory;
  auto* pl = factory.GetReferencePhysList(argv[1]);
  if (!pl) return 3;
  rm->SetUserInitialization(pl);
  rm->SetUserAction(new Gun);
  rm->Initialize();
  rm->BeamOn(0);
  G4EmCalculator calc;
  std::ostringstream out;
  out << std::setprecision(10) << "{\"geant4\": \"" << G4Version << "\", \"physics_list\": \"" << argv[1]
      << "\", \"rows\": [";
  bool first = true;
  for (const auto& name : gMaterials) {
    const G4Material* mat = G4NistManager::Instance()->FindOrBuildMaterial(name);
    const double rho = mat->GetDensity() / (g / cm3);
    for (const char* particle : {"mu-", "mu+"}) {
      const G4ParticleDefinition* pd = G4ParticleTable::GetParticleTable()->FindParticle(particle);
      for (const char* process : {"muIoni", "muPairProd", "muBrems"}) {
        for (double t : energies) {
          const double dedx = calc.ComputeDEDX(t * MeV, pd, process, mat, DBL_MAX) / (MeV / cm) / rho;
          out << (first ? "" : ", ") << "[\"" << name << "\", \"" << particle << "\", \"" << process << "\", " << t
              << ", " << dedx << "]";
          first = false;
        }
      }
    }
  }
  out << "]}\n";
  std::ofstream(argv[3]) << out.str();
  delete rm;
  return 0;
}
