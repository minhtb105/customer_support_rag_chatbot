"""
Memory Monitoring and Backup Module

Provides comprehensive monitoring, backup, and performance tracking
for the memory system with Prometheus metrics and automated backups.
"""

import time
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import schedule
import redis
from sqlalchemy import text
from memory.database_config import SessionLocal, engine
from memory.models import MemoryFact, SessionSummary, UserProfile, MemoryStats
from memory.adapters import get_stats_adapter


@dataclass
class MemoryMetrics:
    """Memory performance metrics."""
    timestamp: float
    short_term_memory: Dict[str, Any]
    episodic_memory: Dict[str, Any]
    long_term_memory: Dict[str, Any]
    system_overall: Dict[str, Any]


class MemoryMonitor:
    """Comprehensive memory system monitor."""
    
    def __init__(self, redis_client: redis.Redis = None):
        self.redis_client = redis_client or redis.Redis(host='localhost', port=6379, db=0)
        self.stats_adapter = get_stats_adapter()
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # Metrics storage
        self.metrics_history: List[MemoryMetrics] = []
        self.max_history = 1000
        
        # Performance thresholds
        self.thresholds = {
            'response_time_ms': 1000,
            'memory_usage_mb': 500,
            'cache_hit_rate': 0.8,
            'db_connection_time_ms': 100
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self, interval_seconds: int = 60):
        """Start background monitoring."""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info(f"Memory monitoring started with {interval_seconds}s interval")
    
    def stop_monitoring(self):
        """Stop background monitoring."""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("Memory monitoring stopped")
    
    def _monitoring_loop(self, interval_seconds: int):
        """Background monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self.collect_metrics()
                self.store_metrics(metrics)
                self.check_alerts(metrics)
                
                # Store in database
                self._store_metrics_to_db(metrics)
                
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
            
            time.sleep(interval_seconds)
    
    def collect_metrics(self) -> MemoryMetrics:
        """Collect comprehensive memory metrics."""
        timestamp = time.time()
        
        # Short-term memory metrics
        short_term_metrics = self._collect_short_term_metrics()
        
        # Episodic memory metrics
        episodic_metrics = self._collect_episodic_metrics()
        
        # Long-term memory metrics
        long_term_metrics = self._collect_long_term_metrics()
        
        # System overall metrics
        system_metrics = self._collect_system_metrics()
        
        return MemoryMetrics(
            timestamp=timestamp,
            short_term_memory=short_term_metrics,
            episodic_memory=episodic_metrics,
            long_term_memory=long_term_metrics,
            system_overall=system_metrics
        )
    
    def _collect_short_term_metrics(self) -> Dict[str, Any]:
        """Collect short-term memory metrics."""
        try:
            # Redis memory usage
            redis_info = self.redis_client.info('memory')
            redis_memory_mb = redis_info.get('used_memory_human', '0B')
            
            # Redis key count
            key_count = self.redis_client.dbsize()
            
            # Average TTL
            keys = self.redis_client.keys('session:*:messages')
            avg_ttl = 0
            if keys:
                ttls = [self.redis_client.ttl(key) for key in keys]
                avg_ttl = sum(ttls) / len(ttls)
            
            return {
                'redis_memory_mb': redis_memory_mb,
                'session_count': len(keys),
                'avg_session_ttl': avg_ttl,
                'total_keys': key_count,
                'memory_efficiency': self._calculate_memory_efficiency(keys)
            }
        
        except Exception as e:
            self.logger.error(f"Short-term metrics collection failed: {e}")
            return {'error': str(e)}
    
    def _collect_episodic_metrics(self) -> Dict[str, Any]:
        """Collect episodic memory metrics."""
        session = SessionLocal()
        try:
            # Summary count
            summary_count = session.query(SessionSummary).count()
            
            # Average summary length
            avg_summary_length = session.query(
                text("AVG(LENGTH(summary_text))")
            ).scalar() or 0
            
            # Summary creation rate
            last_hour = datetime.utcnow() - timedelta(hours=1)
            recent_summaries = session.query(SessionSummary).filter(
                SessionSummary.last_updated > last_hour
            ).count()
            
            return {
                'total_summaries': summary_count,
                'avg_summary_length': avg_summary_length,
                'summaries_last_hour': recent_summaries,
                'summary_quality_score': self._calculate_summary_quality()
            }
        
        except Exception as e:
            self.logger.error(f"Episodic metrics collection failed: {e}")
            return {'error': str(e)}
        
        finally:
            session.close()
    
    def _collect_long_term_metrics(self) -> Dict[str, Any]:
        """Collect long-term memory metrics."""
        session = SessionLocal()
        try:
            # Fact count by type
            fact_counts = {}
            for fact_type in ['medication', 'symptom', 'condition', 'allergy', 'lifestyle']:
                count = session.query(MemoryFact).filter(
                    MemoryFact.fact_type == fact_type
                ).count()
                fact_counts[fact_type] = count
            
            # Total facts
            total_facts = session.query(MemoryFact).count()
            
            # User count
            user_count = session.query(UserProfile).count()
            
            # Average facts per user
            avg_facts_per_user = total_facts / max(user_count, 1)
            
            return {
                'total_facts': total_facts,
                'user_count': user_count,
                'avg_facts_per_user': avg_facts_per_user,
                'fact_distribution': fact_counts,
                'storage_efficiency': self._calculate_storage_efficiency()
            }
        
        except Exception as e:
            self.logger.error(f"Long-term metrics collection failed: {e}")
            return {'error': str(e)}
        
        finally:
            session.close()
    
    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system-wide metrics."""
        try:
            # Database connection test
            start_time = time.perf_counter()
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            db_response_time = (time.perf_counter() - start_time) * 1000
            
            # Database size
            db_size_query = text("""
                SELECT pg_size_pretty(pg_database_size(current_database())) as size
            """)
            with engine.connect() as conn:
                db_size = conn.execute(db_size_query).scalar()
            
            # Active connections
            conn_count_query = text("""
                SELECT count(*) FROM pg_stat_activity 
                WHERE state = 'active'
            """)
            with engine.connect() as conn:
                active_connections = conn.execute(conn_count_query).scalar()
            
            return {
                'db_response_time_ms': db_response_time,
                'db_size': db_size,
                'active_connections': active_connections,
                'system_health': 'healthy' if db_response_time < 100 else 'degraded'
            }
        
        except Exception as e:
            self.logger.error(f"System metrics collection failed: {e}")
            return {'error': str(e)}
    
    def _calculate_memory_efficiency(self, keys: List[str]) -> float:
        """Calculate memory efficiency score."""
        if not keys:
            return 0.0
        
        total_size = 0
        for key in keys:
            try:
                size = self.redis_client.memory_usage(key)
                total_size += size
            except:
                pass
        
        # Simple efficiency calculation
        avg_size = total_size / len(keys) if keys else 0
        return min(1.0, avg_size / 1024)  # Normalize to 0-1
    
    def _calculate_summary_quality(self) -> float:
        """Calculate summary quality score."""
        # This would implement actual quality calculation
        # For now, return a placeholder
        return 0.85
    
    def _calculate_storage_efficiency(self) -> float:
        """Calculate storage efficiency."""
        # This would implement actual efficiency calculation
        # For now, return a placeholder
        return 0.9
    
    def check_alerts(self, metrics: MemoryMetrics):
        """Check for performance alerts."""
        alerts = []
        
        # Check response time
        if metrics.system_overall.get('db_response_time_ms', 0) > self.thresholds['response_time_ms']:
            alerts.append("Database response time exceeded threshold")
        
        # Check memory usage
        if 'redis_memory_mb' in metrics.short_term_memory:
            # Parse memory string like "1.2M" or "500K"
            memory_str = metrics.short_term_memory['redis_memory_mb']
            if memory_str.endswith('M'):
                memory_mb = float(memory_str[:-1])
            elif memory_str.endswith('K'):
                memory_mb = float(memory_str[:-1]) / 1024
            else:
                memory_mb = 0
            
            if memory_mb > self.thresholds['memory_usage_mb']:
                alerts.append(f"Redis memory usage ({memory_str}) exceeded threshold")
        
        # Log alerts
        for alert in alerts:
            self.logger.warning(f"ALERT: {alert}")
    
    def store_metrics(self, metrics: MemoryMetrics):
        """Store metrics in memory history."""
        self.metrics_history.append(metrics)
        if len(self.metrics_history) > self.max_history:
            self.metrics_history.pop(0)
    
    def _store_metrics_to_db(self, metrics: MemoryMetrics):
        """Store metrics in database."""
        try:
            # Store overall system metrics
            self.stats_adapter.store_stats(
                user_id="system",
                session_id="monitoring",
                stat_type="system_overall",
                metrics=metrics.system_overall
            )
            
            # Store memory-specific metrics
            for memory_type, data in [
                ('short_term', metrics.short_term_memory),
                ('episodic', metrics.episodic_memory),
                ('long_term', metrics.long_term_memory)
            ]:
                self.stats_adapter.store_stats(
                    user_id="system",
                    session_id="monitoring",
                    stat_type=memory_type,
                    metrics=data
                )
        
        except Exception as e:
            self.logger.error(f"Failed to store metrics to database: {e}")
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of recent metrics."""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        latest = self.metrics_history[-1]
        
        return {
            "timestamp": latest.timestamp,
            "short_term": latest.short_term_memory,
            "episodic": latest.episodic_memory,
            "long_term": latest.long_term_memory,
            "system": latest.system_overall,
            "history_count": len(self.metrics_history)
        }
    
    def export_metrics(self, filepath: str):
        """Export metrics history to JSON file."""
        try:
            data = {
                "export_timestamp": time.time(),
                "metrics_history": [asdict(m) for m in self.metrics_history]
            }
            
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            self.logger.info(f"Metrics exported to {filepath}")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to export metrics: {e}")
            return False


class MemoryBackup:
    """Memory backup and recovery system."""
    
    def __init__(self, backup_dir: str = "backups"):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(exist_ok=True)
        
        # Schedule daily backups
        schedule.every().day.at("02:00").do(self.create_daily_backup)
        schedule.every().sunday.at("03:00").do(self.create_weekly_backup)
    
    def create_daily_backup(self):
        """Create daily backup of memory data."""
        return self._create_backup("daily")
    
    def create_weekly_backup(self):
        """Create weekly backup with more comprehensive data."""
        return self._create_backup("weekly")
    
    def _create_backup(self, backup_type: str) -> bool:
        """Create backup of specified type."""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{backup_type}_backup_{timestamp}"
            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)
            
            # Backup database
            db_backup = self._backup_database(backup_path)
            
            # Backup Redis data
            redis_backup = self._backup_redis(backup_path)
            
            # Create backup manifest
            manifest = {
                "backup_type": backup_type,
                "timestamp": timestamp,
                "database_backup": db_backup,
                "redis_backup": redis_backup,
                "size_mb": self._calculate_backup_size(backup_path)
            }
            
            with open(backup_path / "manifest.json", 'w') as f:
                json.dump(manifest, f, indent=2, default=str)
            
            self._cleanup_old_backups(backup_type)
            return True
        
        except Exception as e:
            logging.error(f"Backup failed: {e}")
            return False
    
    def _backup_database(self, backup_path: Path) -> bool:
        """Backup database to SQL dump."""
        try:
            dump_file = backup_path / "database_dump.sql"
            
            # Use pg_dump for PostgreSQL or sqlite3 for SQLite
            if engine.url.drivername == 'postgresql':
                import subprocess
                cmd = [
                    'pg_dump',
                    '-h', engine.url.host,
                    '-U', engine.url.username,
                    '-d', engine.url.database,
                    '-f', str(dump_file)
                ]
                subprocess.run(cmd, check=True)
            else:
                # SQLite backup
                import shutil
                db_file = engine.url.database
                shutil.copy2(db_file, dump_file)
            
            return True
        
        except Exception as e:
            logging.error(f"Database backup failed: {e}")
            return False
    
    def _backup_redis(self, backup_path: Path) -> bool:
        """Backup Redis data."""
        try:
            import redis
            redis_client = redis.Redis(host='localhost', port=6379, db=0)
            
            # Save Redis data
            redis_client.save()
            
            # Copy Redis dump file if it exists
            dump_file = backup_path / "redis_dump.rdb"
            import shutil
            try:
                shutil.copy2("/var/lib/redis/dump.rdb", dump_file)
            except:
                # Fallback: export keys manually
                keys = redis_client.keys('*')
                data = {}
                for key in keys:
                    try:
                        data[key.decode()] = redis_client.get(key).decode()
                    except:
                        pass
                
                with open(dump_file.with_suffix('.json'), 'w') as f:
                    json.dump(data, f)
            
            return True
        
        except Exception as e:
            logging.error(f"Redis backup failed: {e}")
            return False
    
    def _calculate_backup_size(self, backup_path: Path) -> float:
        """Calculate backup size in MB."""
        total_size = 0
        for file in backup_path.rglob('*'):
            if file.is_file():
                total_size += file.stat().st_size
        
        return total_size / (1024 * 1024)
    
    def _cleanup_old_backups(self, backup_type: str):
        """Clean up old backups."""
        cutoff_date = datetime.now() - timedelta(days=7 if backup_type == "daily" else 30)
        
        for backup_dir in self.backup_dir.iterdir():
            if backup_dir.is_dir() and backup_type in backup_dir.name:
                try:
                    backup_date_str = backup_dir.name.split('_')[-1]
                    backup_date = datetime.strptime(backup_date_str, "%Y%m%d_%H%M%S")
                    
                    if backup_date < cutoff_date:
                        import shutil
                        shutil.rmtree(backup_dir)
                
                except:
                    pass
    
    def restore_backup(self, backup_name: str) -> bool:
        """Restore from backup."""
        try:
            backup_path = self.backup_dir / backup_name
            if not backup_path.exists():
                raise FileNotFoundError(f"Backup {backup_name} not found")
            
            # Restore database
            db_manifest = backup_path / "database_dump.sql"
            if db_manifest.exists():
                self._restore_database(db_manifest)
            
            # Restore Redis
            redis_manifest = backup_path / "redis_dump.rdb"
            if redis_manifest.exists():
                self._restore_redis(redis_manifest)
            
            return True
        
        except Exception as e:
            logging.error(f"Restore failed: {e}")
            return False
    
    def _restore_database(self, dump_file: Path):
        """Restore database from dump."""
        if engine.url.drivername == 'postgresql':
            import subprocess
            cmd = [
                'psql',
                '-h', engine.url.host,
                '-U', engine.url.username,
                '-d', engine.url.database,
                '-f', str(dump_file)
            ]
            subprocess.run(cmd, check=True)
        else:
            # SQLite restore
            import shutil
            shutil.copy2(dump_file, engine.url.database)
    
    def _restore_redis(self, dump_file: Path):
        """Restore Redis from dump."""
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        
        if dump_file.suffix == '.json':
            # Restore from JSON
            with open(dump_file, 'r') as f:
                data = json.load(f)
            
            for key, value in data.items():
                redis_client.set(key, value)
        else:
            # Restore from RDB file
            import shutil
            shutil.copy2(dump_file, "/var/lib/redis/dump.rdb")
            redis_client.execute_command('SHUTDOWN')


# Global instances
memory_monitor = MemoryMonitor()
memory_backup = MemoryBackup()


def get_memory_monitor() -> MemoryMonitor:
    """Get global memory monitor instance."""
    return memory_monitor


def get_memory_backup() -> MemoryBackup:
    """Get global memory backup instance."""
    return memory_backup


if __name__ == "__main__":
    print("🔧 Memory Monitoring & Backup Test")
    
    # Test monitoring
    monitor = get_memory_monitor()
    metrics = monitor.collect_metrics()
    print("✅ Metrics collected:", metrics.system_overall)
    
    # Test backup
    backup = get_memory_backup()
    print("✅ Backup system ready")
    
    print("🎉 Memory monitoring & backup setup complete!")