// Flowing ASCII Background Generator
class FlowingAsciiBackground {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        // Symbols that flow
        this.symbols = [
            'Δ', '∇', '∂', 'θ', 'λ', 'σ', 'μ', '∞', 'Σ', 'Π', '∫',
            '→', '←', '↑', '↓', '⟶', '⇒', '⊕', '⊗', '∈', '∀', '∃',
            '●', '○', '◆', '◇', '■', '□', '▲', '△',
            '0', '1', 'α', 'β', 'γ', 'ε', 'η', 'ω'
        ];

        this.fontSize = 14;
        this.columns = Math.floor(this.canvas.width / this.fontSize);

        // Initialize drops
        this.drops = [];
        for (let i = 0; i < this.columns; i++) {
            this.drops[i] = {
                y: Math.random() * -100,
                speed: 0.5 + Math.random() * 1,
                symbol: this.randomSymbol()
            };
        }

        // Horizontal data streams
        this.streams = [];
        this.initStreams();

        window.addEventListener('resize', () => this.resize());
        this.animate();
    }

    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
        this.columns = Math.floor(this.canvas.width / this.fontSize);
    }

    initStreams() {
        const streamCount = 3;
        for (let i = 0; i < streamCount; i++) {
            this.streams.push({
                y: (this.canvas.height / streamCount) * i + Math.random() * 100,
                x: -200,
                speed: 1 + Math.random() * 2,
                text: this.generateStreamText()
            });
        }
    }

    generateStreamText() {
        const texts = [
            '[LOGS] → [DISTILL] → [TRAIN] → [ADAPT]',
            'θ′ = θ - α·∇L(θ)',
            'Loss ↘  Performance ↗',
            'W = W_base + ΔW_LoRA',
            '65% → 85% = +31%'
        ];
        return texts[Math.floor(Math.random() * texts.length)];
    }

    randomSymbol() {
        return this.symbols[Math.floor(Math.random() * this.symbols.length)];
    }

    animate() {
        // Semi-transparent black for trail effect
        this.ctx.fillStyle = 'rgba(0, 0, 0, 0.05)';
        this.ctx.fillRect(0, 0, this.canvas.width, this.canvas.height);

        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = `${this.fontSize}px monospace`;

        // Draw falling symbols
        for (let i = 0; i < this.drops.length; i++) {
            const drop = this.drops[i];

            // Random opacity for depth
            const opacity = 0.3 + Math.random() * 0.4;
            this.ctx.fillStyle = `rgba(255, 255, 255, ${opacity})`;

            this.ctx.fillText(
                drop.symbol,
                i * this.fontSize,
                drop.y * this.fontSize
            );

            // Move drop down
            drop.y += drop.speed;

            // Reset drop when it goes off screen
            if (drop.y * this.fontSize > this.canvas.height && Math.random() > 0.95) {
                drop.y = 0;
                drop.speed = 0.5 + Math.random() * 1;
                drop.symbol = this.randomSymbol();
            }
        }

        // Draw horizontal streams
        this.ctx.fillStyle = 'rgba(255, 255, 255, 0.2)';
        for (const stream of this.streams) {
            this.ctx.fillText(stream.text, stream.x, stream.y);

            stream.x += stream.speed;

            // Reset stream when it goes off screen
            if (stream.x > this.canvas.width) {
                stream.x = -200;
                stream.y = Math.random() * this.canvas.height;
                stream.text = this.generateStreamText();
            }
        }

        requestAnimationFrame(() => this.animate());
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
    // Initialize flowing ASCII background
    new FlowingAsciiBackground('ascii-background');

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
