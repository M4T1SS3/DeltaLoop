// Organic Abstract Background for Hero Section
class AbstractHeroBackground {
    constructor(canvasId) {
        this.canvas = document.getElementById(canvasId);
        if (!this.canvas) return;

        this.ctx = this.canvas.getContext('2d');
        this.resize();

        this.time = 0;

        // Layer 1: Mesh Grid
        this.meshRows = 20;
        this.meshCols = 20;

        // Mouse tracking for mesh interaction
        this.mouseX = -1000;
        this.mouseY = -1000;

        window.addEventListener('resize', () => {
            this.resize();
        });

        // Track mouse position relative to canvas
        this.canvas.addEventListener('mousemove', (e) => {
            const rect = this.canvas.getBoundingClientRect();
            this.mouseX = e.clientX - rect.left;
            this.mouseY = e.clientY - rect.top;
        });

        this.canvas.addEventListener('mouseleave', () => {
            this.mouseX = -1000;
            this.mouseY = -1000;
        });

        this.animate();
    }

    resize() {
        // Fill the hero section
        const parent = this.canvas.parentElement;
        this.canvas.width = parent.offsetWidth;
        this.canvas.height = parent.offsetHeight;
    }



    drawMesh() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
        this.ctx.lineWidth = 1;

        const cellWidth = this.canvas.width / this.meshCols;
        const cellHeight = this.canvas.height / this.meshRows;
        const mouseInfluenceRadius = 250;

        // Horizontal lines with mouse interaction only
        for (let row = 0; row <= this.meshRows; row++) {
            this.ctx.beginPath();
            for (let col = 0; col <= this.meshCols; col++) {
                const baseX = col * cellWidth;
                const baseY = row * cellHeight;

                // Mouse-based displacement
                const dx = baseX - this.mouseX;
                const dy = baseY - this.mouseY;
                const distance = Math.sqrt(dx * dx + dy * dy);

                let mouseDisplacementX = 0;
                let mouseDisplacementY = 0;

                if (distance < mouseInfluenceRadius && distance > 0) {
                    // Inverse square falloff for smooth bulge
                    const force = (1 - distance / mouseInfluenceRadius) ** 2;
                    const angle = Math.atan2(dy, dx);
                    mouseDisplacementX = Math.cos(angle) * force * 60;
                    mouseDisplacementY = Math.sin(angle) * force * 60;
                }

                const x = baseX + mouseDisplacementX;
                const y = baseY + mouseDisplacementY;

                if (col === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            this.ctx.stroke();
        }

        // Vertical lines with mouse interaction only
        for (let col = 0; col <= this.meshCols; col++) {
            this.ctx.beginPath();
            for (let row = 0; row <= this.meshRows; row++) {
                const baseX = col * cellWidth;
                const baseY = row * cellHeight;

                // Mouse-based displacement
                const dx = baseX - this.mouseX;
                const dy = baseY - this.mouseY;
                const distance = Math.sqrt(dx * dx + dy * dy);

                let mouseDisplacementX = 0;
                let mouseDisplacementY = 0;

                if (distance < mouseInfluenceRadius && distance > 0) {
                    const force = (1 - distance / mouseInfluenceRadius) ** 2;
                    const angle = Math.atan2(dy, dx);
                    mouseDisplacementX = Math.cos(angle) * force * 60;
                    mouseDisplacementY = Math.sin(angle) * force * 60;
                }

                const x = baseX + mouseDisplacementX;
                const y = baseY + mouseDisplacementY;

                if (row === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            this.ctx.stroke();
        }
    }



    animate() {
        this.time++;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw mesh grid
        this.drawMesh();

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
    // Initialize abstract background
    new AbstractHeroBackground('ascii-background');

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
