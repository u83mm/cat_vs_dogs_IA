"use strict"; 

var $ = function(id) {
	return document.getElementById(id);
}

window.onload = function() {
	/* Selects all elements with the class name 'show_password'
	and adds an event listener to them. It then checks if there are any elements found
	with that class using `if(showPasswordChars.length > 0)`. */
	let showPasswordChars = document.querySelectorAll('.show_password');

	if(showPasswordChars.length > 0) {
		showPasswordChars.forEach(showPasswordChar => {
			showPasswordChar.addEventListener('click', () => {				
				let input = showPasswordChar.parentNode.previousElementSibling.querySelector('input');
				if(input.type == 'password') {
					input.type = 'text';
					showPasswordChar.src = '/images/eye_closed.svg';
				} else {
					input.type = 'password';
					showPasswordChar.src = '/images/eye.svg';
				}
			});
		});
	}
}
