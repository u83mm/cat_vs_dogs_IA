<?php	
	use Application\Database\Connection;									
	
	try {
		$dbcon = new Connection(include DB_CONFIG_FILE);		
	}
	catch(PDOException $e) {
		$error = $e->getMessage();
		$error_msg = "<p>Hay problemas al conectar con la base de datos, revise la configuración 
						de acceso.</p><p>Descripción del error: <span class='error'>$error</span></p>";
		include(SITE_ROOT . "/view/database_error.php");
		exit();
	}	
?>
