<?php		
	use Application\model\classes\Loader;
	
	$loader = new Loader();
	$loader->init($_SERVER['DOCUMENT_ROOT'] . "/..");

	/** Define site root and URL */
	define("SITE_ROOT", $_SERVER['DOCUMENT_ROOT']);
	define('URL', $_SERVER['REQUEST_URI']);				
	
	/** Define connection */
	define('DB_CONFIG_FILE', SITE_ROOT . '/../Application/Core/db.config.php');	
	require_once(SITE_ROOT . "/../Application/Core/connect.php");
	define('DB_CON', $dbcon);		
?>
