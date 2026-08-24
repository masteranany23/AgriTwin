export type TwinState = 'current' | 'assimilated' | 'forecast';

export const demoField = {
  name: 'Demonstration Field',
  crop: 'Wheat',
  day: 42,
  current: { lai: '2.84', moisture: '31%', biomass: '4.21 t/ha', dvs: '0.74' },
  observation: { source: 'Sentinel-2', lai: '2.61', confidence: '91%' },
  assimilated: { lai: '2.67' },
  forecast: { lai: '3.42', yield: '4.80 t/ha', confidence: '87%' },
} as const;

export const trajectoryData = [
  { day: 1, openLoop: 0.64, assimilated: 0.66, observation: null },
  { day: 7, openLoop: 0.91, assimilated: 0.94, observation: null },
  { day: 14, openLoop: 1.29, assimilated: 1.25, observation: 1.22 },
  { day: 21, openLoop: 1.72, assimilated: 1.64, observation: null },
  { day: 28, openLoop: 2.2, assimilated: 2.08, observation: 2.01 },
  { day: 35, openLoop: 2.58, assimilated: 2.47, observation: null },
  { day: 42, openLoop: 2.84, assimilated: 2.67, observation: 2.61 },
  { day: 49, openLoop: 3.12, assimilated: 3.04, observation: null },
  { day: 56, openLoop: 3.36, assimilated: 3.31, observation: null },
  { day: 63, openLoop: 3.51, assimilated: 3.42, observation: null },
];

export const assimilationSteps = [
  { id: '01', title: 'Current Field', detail: 'Physics-based state' },
  { id: '02', title: 'Observation', detail: 'Sentinel-2 signal' },
  { id: '03', title: 'Data Fusion', detail: 'Uncertainty-weighted' },
  { id: '04', title: 'EnKF Process', detail: 'Ensemble update' },
  { id: '05', title: 'Updated Twin', detail: 'Best estimate' },
  { id: '06', title: 'Future Forecast', detail: 'Projected trajectory' },
] as const;

export const fieldPlants = Array.from({ length: 9 }, (_, row) =>
  Array.from({ length: 17 }, (_, column) => {
    const wave = (row * 7 + column * 11) % 13;
    return {
      row,
      column,
      x: 88 + column * 35 + (row % 2) * 6,
      y: 55 + row * 29,
      height: 12 + (wave % 5),
      opacity: 0.38 + (wave % 6) * 0.075,
      lean: ((wave % 5) - 2) * 1.8,
    };
  }),
).flat();
