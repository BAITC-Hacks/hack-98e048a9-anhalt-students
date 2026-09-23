const fs = require('fs');
const vm = require('vm');

const html = fs.readFileSync('backend/static/index.html', 'utf8');
const scriptMatches = [...html.matchAll(/<script(?:\s+[^>]*)?>([\s\S]*?)<\/script>/gi)];
const inlineScripts = scriptMatches.map(m => m[1].trim()).filter(Boolean);

console.log('Total inline script blocks:', inlineScripts.length);
if (inlineScripts.length !== 1) {
  console.error('ERROR: Expected exactly 1 inline script block, found:', inlineScripts.length);
  process.exit(1);
}

const code = inlineScripts[0];
try {
  new vm.Script(code);
  console.log('SUCCESS: Script parsed cleanly with zero syntax errors! (' + code.length + ' chars)');
  const requiredFns = [
    'openUploadModal', 'closeUploadModal', 'submitScadaUpload', 'onFileSelected',
    'toggleSimulationPlay', 'playFromDayOne', 'startSimulation', 'stopSimulation', 'onSliderChange',
    'stepSimulation', 'updateSimDayView', 'runAgentForecast', 'renderDashboard',
    'openPresentationModal', 'closePresentationModal', 'goToSlide', 'demoStormCutout'
  ];
  for (const fn of requiredFns) {
    if (!code.includes(`function ${fn}`)) {
      console.error(`ERROR: Missing function ${fn}`);
      process.exit(1);
    }
  }
  console.log('SUCCESS: All required UI functions are defined in the script!');
} catch (err) {
  console.error('SYNTAX ERROR:', err);
  process.exit(1);
}
