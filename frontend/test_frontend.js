/**
 * Frontend integration test
 * Tests that all components are properly structured
 */

import { readFileSync } from 'fs';
import { join } from 'path';

function testComponentExists(name) {
    const path = join('src', 'components', `${name}.jsx`);
    try {
        const content = readFileSync(path, 'utf8');
        if (content.includes('export default') || content.includes('export')) {
            console.log(`✓ ${name} component exists and exports`);
            return true;
        }
    } catch (e) {
        console.log(`✗ ${name} component missing`);
        return false;
    }
    return false;
}

function testServiceExists(name) {
    const path = join('src', 'services', `${name}.js`);
    try {
        const content = readFileSync(path, 'utf8');
        if (content.includes('export')) {
            console.log(`✓ ${name} service exists and exports`);
            return true;
        }
    } catch (e) {
        console.log(`✗ ${name} service missing`);
        return false;
    }
    return false;
}

function testBuildOutput() {
    try {
        const indexPath = join('dist', 'index.html');
        readFileSync(indexPath, 'utf8');
        console.log('✓ Build output exists (dist/index.html)');
        return true;
    } catch (e) {
        console.log('✗ Build output missing');
        return false;
    }
}

console.log('='.repeat(60));
console.log('Frontend Integration Tests');
console.log('='.repeat(60));

let passed = 0;
let total = 0;

// Test components
const components = ['LiveInference', 'TrainingMonitor', 'MetricsChart', 'ModelComparison', 'DatasetStats'];
components.forEach(comp => {
    total++;
    if (testComponentExists(comp)) passed++;
});

// Test services
const services = ['api', 'websocket'];
services.forEach(svc => {
    total++;
    if (testServiceExists(svc)) passed++;
});

// Test build
total++;
if (testBuildOutput()) passed++;

console.log('='.repeat(60));
console.log(`Frontend Tests: ${passed}/${total} passed`);
console.log('='.repeat(60));

process.exit(passed === total ? 0 : 1);
