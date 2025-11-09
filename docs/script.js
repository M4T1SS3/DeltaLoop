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

        // Layer 2: Gradient Blobs
        this.blobs = [];
        this.initBlobs();

        // Layer 3: Neural Network
        this.nodes = [];
        this.initNodes();

        window.addEventListener('resize', () => this.resize());
        this.animate();
    }

    resize() {
        this.canvas.width = this.canvas.offsetWidth;
        this.canvas.height = this.canvas.offsetHeight;
    }

    initBlobs() {
        const blobCount = 4;
        for (let i = 0; i < blobCount; i++) {
            this.blobs.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                baseSize: 200 + Math.random() * 200,
                speed: 0.0003 + Math.random() * 0.0002,
                phase: Math.random() * Math.PI * 2,
                driftSpeedX: (Math.random() - 0.5) * 0.2,
                driftSpeedY: (Math.random() - 0.5) * 0.2
            });
        }
    }

    initNodes() {
        const nodeCount = 50;
        for (let i = 0; i < nodeCount; i++) {
            this.nodes.push({
                x: Math.random() * this.canvas.width,
                y: Math.random() * this.canvas.height,
                vx: (Math.random() - 0.5) * 0.3,
                vy: (Math.random() - 0.5) * 0.3,
                radius: 2 + Math.random() * 3,
                phase: Math.random() * Math.PI * 2
            });
        }
    }

    drawMesh() {
        this.ctx.strokeStyle = 'rgba(255, 255, 255, 0.08)';
        this.ctx.lineWidth = 1;

        const cellWidth = this.canvas.width / this.meshCols;
        const cellHeight = this.canvas.height / this.meshRows;

        // Horizontal lines with wave distortion
        for (let row = 0; row <= this.meshRows; row++) {
            this.ctx.beginPath();
            for (let col = 0; col <= this.meshCols; col++) {
                const x = col * cellWidth;
                const baseY = row * cellHeight;

                // Wave distortion
                const wave1 = Math.sin(this.time * 0.5 + col * 0.3) * 15;
                const wave2 = Math.cos(this.time * 0.3 + row * 0.2) * 10;
                const y = baseY + wave1 + wave2;

                if (col === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            this.ctx.stroke();
        }

        // Vertical lines with wave distortion
        for (let col = 0; col <= this.meshCols; col++) {
            this.ctx.beginPath();
            for (let row = 0; row <= this.meshRows; row++) {
                const baseX = col * cellWidth;
                const y = row * cellHeight;

                // Wave distortion
                const wave1 = Math.sin(this.time * 0.4 + row * 0.3) * 15;
                const wave2 = Math.cos(this.time * 0.6 + col * 0.2) * 10;
                const x = baseX + wave1 + wave2;

                if (row === 0) {
                    this.ctx.moveTo(x, y);
                } else {
                    this.ctx.lineTo(x, y);
                }
            }
            this.ctx.stroke();
        }
    }

    drawGradientBlobs() {
        for (const blob of this.blobs) {
            // Organic size pulsing
            const sizePulse = Math.sin(this.time * blob.speed + blob.phase) * 0.3 + 1;
            const currentSize = blob.baseSize * sizePulse;

            // Organic position drifting
            blob.x += blob.driftSpeedX;
            blob.y += blob.driftSpeedY;

            // Wrap around screen
            if (blob.x < -currentSize) blob.x = this.canvas.width + currentSize;
            if (blob.x > this.canvas.width + currentSize) blob.x = -currentSize;
            if (blob.y < -currentSize) blob.y = this.canvas.height + currentSize;
            if (blob.y > this.canvas.height + currentSize) blob.y = -currentSize;

            // Create radial gradient
            const gradient = this.ctx.createRadialGradient(
                blob.x, blob.y, 0,
                blob.x, blob.y, currentSize
            );
            gradient.addColorStop(0, 'rgba(255, 255, 255, 0.25)');
            gradient.addColorStop(0.5, 'rgba(255, 255, 255, 0.1)');
            gradient.addColorStop(1, 'rgba(255, 255, 255, 0)');

            // Draw blob with blur effect
            this.ctx.filter = 'blur(20px)';
            this.ctx.fillStyle = gradient;
            this.ctx.beginPath();
            this.ctx.arc(blob.x, blob.y, currentSize, 0, Math.PI * 2);
            this.ctx.fill();
            this.ctx.filter = 'none';
        }
    }

    drawNeuralNetwork() {
        const maxDistance = 150;

        // Update node positions with organic movement
        for (const node of this.nodes) {
            // Add sine wave variation for organic feel
            const waveX = Math.sin(this.time * 0.001 + node.phase) * 0.5;
            const waveY = Math.cos(this.time * 0.001 + node.phase * 1.5) * 0.5;

            node.x += node.vx + waveX;
            node.y += node.vy + waveY;

            // Wrap around screen
            if (node.x < 0) node.x = this.canvas.width;
            if (node.x > this.canvas.width) node.x = 0;
            if (node.y < 0) node.y = this.canvas.height;
            if (node.y > this.canvas.height) node.y = 0;
        }

        // Draw connections
        for (let i = 0; i < this.nodes.length; i++) {
            for (let j = i + 1; j < this.nodes.length; j++) {
                const dx = this.nodes[i].x - this.nodes[j].x;
                const dy = this.nodes[i].y - this.nodes[j].y;
                const distance = Math.sqrt(dx * dx + dy * dy);

                if (distance < maxDistance) {
                    const opacity = (1 - distance / maxDistance) * 0.3;
                    this.ctx.strokeStyle = `rgba(255, 255, 255, ${opacity})`;
                    this.ctx.lineWidth = 1;
                    this.ctx.beginPath();
                    this.ctx.moveTo(this.nodes[i].x, this.nodes[i].y);
                    this.ctx.lineTo(this.nodes[j].x, this.nodes[j].y);
                    this.ctx.stroke();
                }
            }
        }

        // Draw nodes
        for (const node of this.nodes) {
            // Pulsing opacity
            const pulse = Math.sin(this.time * 0.002 + node.phase) * 0.2 + 0.6;
            this.ctx.fillStyle = `rgba(255, 255, 255, ${pulse})`;
            this.ctx.beginPath();
            this.ctx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
            this.ctx.fill();
        }
    }

    animate() {
        this.time++;

        // Clear canvas
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

        // Draw all layers
        this.drawMesh();
        this.drawGradientBlobs();
        this.drawNeuralNetwork();

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

// Add subtle parallax effect to hero background
window.addEventListener('scroll', () => {
    const scrolled = window.pageYOffset;
    const canvas = document.querySelector('#ascii-background');
    if (canvas && scrolled < window.innerHeight) {
        canvas.style.transform = `translateY(${scrolled * 0.3}px)`;
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
