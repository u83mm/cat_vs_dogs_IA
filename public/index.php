<?php
	declare(strict_types=1);
	
	use Application\Core\App;
	
	require_once($_SERVER['DOCUMENT_ROOT'] . "/../Application/Core/aplication_fns.php");		
	
	$app = new App;
	$app->loadController();	
?>
