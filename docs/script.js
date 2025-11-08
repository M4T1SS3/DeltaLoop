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
