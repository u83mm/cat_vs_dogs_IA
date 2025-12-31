<?php
	namespace Application\Database;
	
	use Exception;
	use PDO;
	use PDOException;
	
	class Connection {
		const ERROR_UNABLE = 'ERROR: Unable to create database connection';
		
		public PDO $pdo;
		
		public function __construct(array $config) {
			if(!isset($config['driver'])) {
				$message = __METHOD__ . ' : ' . self::ERROR_UNABLE;
				
				throw new Exception($message);
			}
			
			$dsn = $config['driver'] . ':host=' . $config['host'] . ';dbname=' . $config['dbname'] . ';charset=' . $config['charset'];
			
			try {
				$this->pdo = new PDO($dsn, $config['user'], $config['password'], $config['errmode']);
			}
			catch(PDOException $e) {
				error_log($e->getMessage());	
			}
		}
	}
?>
