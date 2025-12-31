<?php
	$options = array(PDO::ATTR_ERRMODE => PDO::ERRMODE_EXCEPTION,
						  PDO::ATTR_DEFAULT_FETCH_MODE => PDO::FETCH_ASSOC,
						  PDO::ATTR_EMULATE_PREPARES   => false);
	
	return [
		'driver'	=>	'mysql',
		'host'		=>	'db',
		'dbname'	=>	'my_database',
		'user'		=>	'admin',
		'password'	=>	'admin',
		'charset'	=>	'utf8mb4',
		'errmode'	=> $options
	];
?>
