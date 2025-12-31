<?php	
	require_once($_SERVER['DOCUMENT_ROOT'] . "/../Application/model/classes/Loader.php");
	require_once($_SERVER['DOCUMENT_ROOT'] . "/../vendor/autoload.php");
	require_once($_SERVER['DOCUMENT_ROOT'] . "/../Application/Core/app_config.php");	
	require_once($_SERVER['DOCUMENT_ROOT'] . "/../Application/Core/Controller.php");
	require_once($_SERVER['DOCUMENT_ROOT'] . "/../Application/Core/Model.php");	
	require_once($_SERVER['DOCUMENT_ROOT'] . "/../Application/Core/App.php");
	
	# Clean Apache acces.log and error.log
	require_once(SITE_ROOT . "/../Application/Cron_jobs/clean_access_log.php");  
	require_once(SITE_ROOT . "/../Application/Cron_jobs/clean_error_log.php");
	require_once(SITE_ROOT . "/../Application/Cron_jobs/clean_xdebug_log.php"); 		
?>
