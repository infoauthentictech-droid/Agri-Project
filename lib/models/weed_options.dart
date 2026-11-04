/// Validated groundnut weed species accepted by the recommendation engine.
///
/// Mirrors `COMMON_WEED_SPECIES` in
/// `herbicide/schemas/herbicide_schema.py` — a recommendation is returned for
/// every entry below (verified against the engine).
const List<String> kWeedSpeciesOptions = [
  'Echinochloa colona',
  'Echinochloa colonum',
  'Cyperus difformis',
  'Cyperus rotundus',
  'Commelina benghalensis',
  'Trianthema portulacastrum',
  'Amaranthus viridis',
  'Parthenium hysterophorus',
  'Euphorbia hirta',
  'Euphorbia geniculata',
  'Digera arvensis',
  'Acalypha spp.',
  'Digitaria sanguinalis',
  'Digitaria longiflora',
  'Eleusine indica',
  'Dactyloctenium aegyptium',
  'Cynodon dactylon',
  'Bracharia reptans',
  'Dinebra retroflexa',
  'Dinebra arabica',
];

/// Validated weed categories accepted by the recommendation engine
/// (herbicide/schemas/herbicide_schema.py — WeedCategory).
const List<String> kWeedCategoryOptions = [
  'grassy_weeds',
  'broadleaf_weeds',
  'sedges',
  'grasses_broadleaves',
  'grasses_broadleaves_sedges',
  'monocot_weeds',
];

/// Returns true when [name] is one of the validated knowledge-base species
/// (as opposed to detector placeholder names such as class_0/class_1 or
/// generic descriptions such as "weed").
bool isKnownSpeciesName(String? name) {
  if (name == null || name.trim().isEmpty) return false;
  final n = name.trim().toLowerCase();
  if (RegExp(r'^class_\d+$').hasMatch(n)) return false;
  return kWeedSpeciesOptions.any((s) => s.toLowerCase() == n);
}