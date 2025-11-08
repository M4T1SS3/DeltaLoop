// ASCII Art Generator for Hero Section
class AsciiArtGenerator {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        this.width = 70;
        this.height = 5;

        // Mathematical and AI symbols
        this.symbols = ['Δ', '∇', '∂', 'θ', 'λ', 'σ', 'μ', '∞', 'Σ', 'Π', '∫', '→', '⟶', '⇒', '⊕', '⊗', '∈', '∀', '∃'];
        this.nodeSymbols = ['●', '○', '◆', '◇', '■', '□', '▲', '△'];
        this.connectSymbols = ['─', '│', '┌', '┐', '└', '┘', '├', '┤', '┬', '┴', '┼', '═', '║', '╔', '╗', '╚', '╝'];

        this.frame = 0;
        this.patterns = [
            this.generateNeuralNet.bind(this),
            this.generateDataFlow.bind(this),
            this.generateMathEquation.bind(this),
            this.generateGradientDescent.bind(this)
        ];

        this.currentPattern = 0;
        this.init();
    }

    init() {
        this.render();
        // Change pattern every 5 seconds
        setInterval(() => {
            this.currentPattern = (this.currentPattern + 1) % this.patterns.length;
            this.frame = 0;
        }, 5000);

        // Animate current pattern
        setInterval(() => {
            this.frame++;
            this.render();
        }, 200);
    }

    render() {
        const pattern = this.patterns[this.currentPattern]();
        this.canvas.textContent = pattern;
    }

    generateNeuralNet() {
        let art = '    ╔══════════════════════════════════════════════════════════════════╗\n';

        // Layer 1 (input)
        const nodes1 = this.frame % 2 === 0 ? '●' : '○';
        art += `    ║  ${nodes1}   ${nodes1}   ${nodes1}   ${nodes1}                                                ║\n`;

        // Connections
        const conn = this.frame % 2 === 0 ? '━' : '─';
        art += `    ║   ╲ │ ╱ ╲ │ ╱     ${conn}${conn}→ [DISTILL] ${conn}${conn}→ [TRAIN] ${conn}${conn}→ [ADAPT] ${conn}${conn}→ ∞  ║\n`;

        // Layer 2 (hidden)
        const nodes2 = this.frame % 2 === 0 ? '◆' : '◇';
        art += `    ║    ${nodes2}   ${nodes2}   ${nodes2}      Δθ = -∇L(θ)  ·  learning_rate              ║\n`;

        art += '    ╚══════════════════════════════════════════════════════════════════╝';
        return art;
    }

    generateDataFlow() {
        const symbols = ['⟶', '→', '⇒'];
        const arrow = symbols[this.frame % symbols.length];

        let art = '    ╔══════════════════════════════════════════════════════════════════╗\n';
        art += `    ║  [LOGS] ${arrow} {filter, dedupe} ${arrow} [DATASET] ${arrow} {LoRA} ${arrow} [MODEL]  ║\n`;

        const pulse = this.frame % 3;
        const nodes = pulse === 0 ? '●○○' : pulse === 1 ? '○●○' : '○○●';
        art += `    ║                                                                  ║\n`;
        art += `    ║    ${nodes}  Training: ${this.frame % 500}/500 steps  Loss: ${(2.5 - (this.frame * 0.01) % 2).toFixed(2)}     ║\n`;
        art += '    ╚══════════════════════════════════════════════════════════════════╝';
        return art;
    }

    generateMathEquation() {
        const equations = [
            'L(θ) = Σ log P(y|x,θ)  →  min Loss, max Learning',
            'θ′ = θ - α·∇L(θ)  →  Gradient Descent in Action',
            'W_adapted = W_base + ΔW_LoRA  →  17MB of Pure Knowledge',
            'Performance: 65% → 85% = +31% via Continuous Learning'
        ];

        const eq = equations[Math.floor(this.frame / 5) % equations.length];

        let art = '    ╔══════════════════════════════════════════════════════════════════╗\n';
        art += '    ║                                                                  ║\n';
        art += `    ║    ${eq.padEnd(62)}║\n`;
        art += '    ║                                                                  ║\n';
        art += '    ╚══════════════════════════════════════════════════════════════════╝';
        return art;
    }

    generateGradientDescent() {
        const step = this.frame % 10;
        const dots = '●'.repeat(step) + '○'.repeat(10 - step);

        let art = '    ╔══════════════════════════════════════════════════════════════════╗\n';
        art += `    ║  Optimization: [${dots}] ${step * 10}%                          ║\n`;
        art += '    ║                                                                  ║\n';

        // Gradient visualization
        const descent = this.frame % 20 < 10
            ? '    ║  Loss ↘  ∇θ → ∂L/∂θ → θ_new → Converging to optimal...          ║'
            : '    ║  Model ↗  Performance improving → 65% → 75% → 85% → ∞          ║';

        art += descent + '\n';
        art += '    ╚══════════════════════════════════════════════════════════════════╝';
        return art;
    }
}

// Intersection Observer for scroll animations
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
        }
    });
}, observerOptions);

// Observe all animatable elements
document.addEventListener('DOMContentLoaded', () => {
    // Initialize ASCII art generator
    new AsciiArtGenerator('ascii-canvas');

    // Observe pipeline steps
    const pipelineSteps = document.querySelectorAll('.pipeline-step');
    pipelineSteps.forEach(step => observer.observe(step));

    // Observe metric cards
    const metricCards = document.querySelectorAll('.metric-card');
    metricCards.forEach(card => observer.observe(card));

    // Observe comparison cards
    const comparisonCards = document.querySelectorAll('.comparison-card');
    comparisonCards.forEach(card => observer.observe(card));

    // Animate numbers on scroll
    animateMetrics();
});

// Animate metric numbers
function animateMetrics() {
    const metricCards = document.querySelectorAll('.metric-card');

    metricCards.forEach(card => {
        const metricChange = card.querySelector('.metric-change');
        const targetText = metricChange.textContent;

        // Extract number and sign
        const match = targetText.match(/([+-]?\d+)/);
        if (!match) return;

        const targetNum = parseInt(match[1]);
        const isPositive = targetText.includes('+');
        const isNegative = targetText.includes('-');
        const hasPercent = targetText.includes('%');

        // Reset to 0
        metricChange.textContent = '0' + (hasPercent ? '%' : '');

        // Create observer for this card
        const cardObserver = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting && !card.dataset.animated) {
                    card.dataset.animated = 'true';
                    animateNumber(metricChange, targetNum, isPositive, isNegative, hasPercent);
                    cardObserver.unobserve(card);
                }
            });
        }, { threshold: 0.5 });

        cardObserver.observe(card);
    });
}

function animateNumber(element, target, isPositive, isNegative, hasPercent) {
    const duration = 1500;
    const start = 0;
    const startTime = performance.now();

    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (easeOutQuart)
        const eased = 1 - Math.pow(1 - progress, 4);
        const current = Math.floor(eased * Math.abs(target));

        let displayValue = current.toString();
        if (isPositive) displayValue = '+' + displayValue;
        if (isNegative) displayValue = '-' + displayValue;
        if (hasPercent) displayValue += '%';

        element.textContent = displayValue;

        if (progress < 1) {
            requestAnimationFrame(update);
        } else {
            // Set final value
            let finalValue = Math.abs(target).toString();
            if (isPositive) finalValue = '+' + finalValue;
            if (isNegative) finalValue = '-' + finalValue;
            if (hasPercent) finalValue += '%';
            element.textContent = finalValue;
        }
    }

    requestAnimationFrame(update);
}

// Smooth scroll for anchor links
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener('click', function (e) {
        e.preventDefault();
        const target = document.querySelector(this.getAttribute('href'));
        if (target) {
            target.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// Add subtle parallax effect to hero grid
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const grid = document.querySelector('.grid-background');
    if (grid && scrolled < window.innerHeight) {
        grid.style.transform = `translateY(${scrolled * 0.5}px)`;
    }
});

// Copy install command on click
const installCommand = document.querySelector('.install-command code');
if (installCommand) {
    installCommand.style.cursor = 'pointer';
    installCommand.title = 'Click to copy';

    installCommand.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText('pip install deltaloop');

            // Show feedback
            const originalText = installCommand.textContent;
            installCommand.textContent = 'Copied! ✓';

            setTimeout(() => {
                installCommand.textContent = originalText;
            }, 2000);
        } catch (err) {
            console.error('Failed to copy:', err);
        }
    });
}
