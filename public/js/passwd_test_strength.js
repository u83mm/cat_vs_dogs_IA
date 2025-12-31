"use strict";

const passwordInput = document.getElementById('password');
const strengthBar = document.getElementById('strength_bar');
const message = document.getElementById('message');

passwordInput.addEventListener('keyup', function() {  
    const strength = getPasswordStrength(this.value);

    strengthBar.className = 'strength_bar ' + strength.class;
    message.textContent = strength.message;
});

function getPasswordStrength(password) {
    let strength = {
        class: '',
        message: ''
    };

    if(!password) {
        strength.class = 'none';
        strength.message = '';
    } else if (password.length < 8) {
        strength.class = 'weak';
        strength.message = 'Too short';
    } else {
        let score = 0;

        if (/[a-z]/.test(password)) score++;
        if (/[A-Z]/.test(password)) score++;
        if (/[0-9]/.test(password)) score++;
        if (/[^a-zA-Z0-9]/.test(password) && password.length >= 12) score++;        

        switch (score) {
            case 1:
                strength.class = 'weak';
                strength.message = 'Weak';
                break;
            case 2:
                strength.class = 'medium';
                strength.message = 'Medium';
                break;
            case 3:
                strength.class = 'strong';
                strength.message = 'Strong';
                break;
            case 4:
                strength.class = 'very_strong';
                strength.message = 'Very Strong';
                break;
        }
    }

    return strength;
}