"use strict"; 

/** Función que crea objeto XMLHttpRequest para todos los navegadores */

function getXMLHTTPRequest() {
	var peticion = false;
	try {
		/* for Firefox */
		peticion = new XMLHttpRequest();
	} catch (err) {
		try {
			/* for some versions of IE */
			peticion = new ActiveXObject("Msxml2.XMLHTTP");
		} catch (err) {
			try {
				/* for some other versions of IE */
				peticion = new ActiveXObject("Microsoft.XMLHTTP");
			} catch (err) {
				peticion = false;
			}
		}
	}
	return peticion;
}