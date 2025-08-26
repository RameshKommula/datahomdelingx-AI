"""Command-line interface for DataModeling-AI."""

import typer
from typing import List, Optional
from pathlib import Path
import json
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich import print as rprint
from loguru import logger
import sys

from .config import init_config, get_config
from .engine.analysis_engine import AnalysisEngine, AnalysisRequest

app = typer.Typer(
    name="datamodeling-ai",
    help="Automate data modeling with AI agents for Hive, Trino, and Presto data lakes.",
    rich_markup_mode="rich"
)

console = Console()


@app.command()
def analyze(
    connection_type: str = typer.Argument(..., help="Connection type: hive, trino, or presto"),
    database: str = typer.Argument(..., help="Database name to analyze"),
    tables: Optional[List[str]] = typer.Option(None, "--table", "-t", help="Specific tables to analyze (default: all)"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="Output file path (JSON format)"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path"),
    no_profiling: bool = typer.Option(False, "--no-profiling", help="Skip data profiling"),
    no_ai: bool = typer.Option(False, "--no-ai", help="Skip AI recommendations"),
    sample_size: int = typer.Option(10000, "--sample-size", "-s", help="Sample size for data profiling"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json or yaml"),
    workflow: Optional[List[str]] = typer.Option(None, "--workflow", "-w", help="Custom workflow stages: discovery, analysis, modeling, optimization"),
    parallel: Optional[List[str]] = typer.Option(None, "--parallel", "-p", help="Stages to run in parallel"),
    workflow_type: Optional[str] = typer.Option(None, "--workflow-type", help="Predefined workflow: discovery, quality, modeling, optimization"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """Analyze a database and generate data modeling recommendations."""
    
    # Configure logging
    if verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")
    
    # Initialize configuration
    if config_file:
        init_config(str(config_file))
    
    try:
        # Create analysis request
        request = AnalysisRequest(
            connection_type=connection_type,
            database=database,
            tables=tables,
            include_data_profiling=not no_profiling,
            include_ai_recommendations=not no_ai,
            sample_size=sample_size
        )
        
        # Create engine and perform analysis
        engine = AnalysisEngine()
        
        # Determine analysis method based on parameters
        if workflow_type:
            analysis_method = getattr(engine, f"analyze_{workflow_type}_focused", None)
            if not analysis_method:
                rprint(f"[red]Unknown workflow type: {workflow_type}[/red]")
                rprint("Available types: discovery, quality, modeling, optimization")
                raise typer.Exit(1)
        elif workflow:
            analysis_method = lambda req: engine.analyze_with_custom_workflow(req, workflow, parallel)
        else:
            analysis_method = engine.analyze
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if workflow or workflow_type:
                task = progress.add_task("Executing AI workflow...", total=None)
            else:
                task = progress.add_task("Analyzing database...", total=None)
            result = analysis_method(request)
        
        if not result.success:
            rprint(f"[red]Analysis failed: {result.error_message}[/red]")
            raise typer.Exit(1)
        
        # Display results
        display_analysis_results(result)
        
        # Save to file if requested
        if output:
            save_results(result, output, format)
            rprint(f"[green]Results saved to {output}[/green]")
        
    except Exception as e:
        rprint(f"[red]Error: {e}[/red]")
        if verbose:
            logger.exception("Analysis failed")
        raise typer.Exit(1)


@app.command()
def test_connections():
    """Test connections to all configured data lakes."""
    try:
        engine = AnalysisEngine()
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("Testing connections...", total=None)
            results = engine.test_connections()
        
        # Display results in a table
        table = Table(title="Connection Test Results")
        table.add_column("Connection Type", style="cyan")
        table.add_column("Status", style="bold")
        
        for conn_type, success in results.items():
            status = "[green]✓ Connected[/green]" if success else "[red]✗ Failed[/red]"
            table.add_row(conn_type.upper(), status)
        
        console.print(table)
        
    except Exception as e:
        rprint(f"[red]Error testing connections: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_databases(
    connection_type: str = typer.Argument(..., help="Connection type: hive, trino, or presto"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path")
):
    """List available databases for a connection type."""
    if config_file:
        init_config(str(config_file))
    
    try:
        engine = AnalysisEngine()
        databases = engine.get_available_databases(connection_type)
        
        if databases:
            rprint(f"[cyan]Available databases in {connection_type.upper()}:[/cyan]")
            for db in databases:
                rprint(f"  • {db}")
        else:
            rprint(f"[yellow]No databases found or connection failed for {connection_type}[/yellow]")
            
    except Exception as e:
        rprint(f"[red]Error listing databases: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def list_tables(
    connection_type: str = typer.Argument(..., help="Connection type: hive, trino, or presto"),
    database: str = typer.Argument(..., help="Database name"),
    config_file: Optional[Path] = typer.Option(None, "--config", "-c", help="Configuration file path")
):
    """List available tables in a database."""
    if config_file:
        init_config(str(config_file))
    
    try:
        engine = AnalysisEngine()
        tables = engine.get_available_tables(connection_type, database)
        
        if tables:
            rprint(f"[cyan]Available tables in {connection_type.upper()}.{database}:[/cyan]")
            for table in tables:
                rprint(f"  • {table}")
        else:
            rprint(f"[yellow]No tables found in {connection_type}.{database}[/yellow]")
            
    except Exception as e:
        rprint(f"[red]Error listing tables: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def workflow_info():
    """Show information about available AI agent workflows."""
    rprint("[bold cyan]DataModeling-AI Workflow Information[/bold cyan]")
    rprint()
    
    # Available agents
    rprint("[bold]Available AI Agents:[/bold]")
    rprint("• [cyan]Discovery Agent[/cyan] - Discovers and catalogs data assets")
    rprint("• [cyan]Analysis Agent[/cyan] - Performs deep data quality analysis")
    rprint("• [cyan]Modeling Agent[/cyan] - Provides dimensional modeling recommendations")
    rprint("• [cyan]Optimization Agent[/cyan] - Suggests performance optimizations")
    rprint()
    
    # Workflow stages
    rprint("[bold]Workflow Stages:[/bold]")
    rprint("• [green]discovery[/green] - Data asset discovery and cataloging")
    rprint("• [green]analysis[/green] - Data quality and pattern analysis")
    rprint("• [green]modeling[/green] - Data modeling recommendations")
    rprint("• [green]optimization[/green] - Performance optimization suggestions")
    rprint()
    
    # Predefined workflows
    rprint("[bold]Predefined Workflows:[/bold]")
    rprint("• [yellow]discovery[/yellow] - Discovery only (fast)")
    rprint("• [yellow]quality[/yellow] - Discovery + Analysis (data quality focused)")
    rprint("• [yellow]modeling[/yellow] - Discovery + Modeling (schema design focused)")
    rprint("• [yellow]optimization[/yellow] - Discovery + Analysis + Optimization (performance focused)")
    rprint()
    
    # Usage examples
    rprint("[bold]Usage Examples:[/bold]")
    rprint("# Default workflow (all stages)")
    rprint("[dim]python main.py analyze hive my_db[/dim]")
    rprint()
    rprint("# Discovery only")
    rprint("[dim]python main.py analyze hive my_db --workflow-type discovery[/dim]")
    rprint()
    rprint("# Custom workflow")
    rprint("[dim]python main.py analyze hive my_db --workflow discovery analysis --parallel analysis[/dim]")
    rprint()
    rprint("# Quality-focused analysis")
    rprint("[dim]python main.py analyze hive my_db --workflow-type quality[/dim]")


@app.command()
def init_config(
    output_path: Path = typer.Option("config.yaml", "--output", "-o", help="Output configuration file path")
):
    """Initialize a new configuration file."""
    try:
        # Copy the example config
        example_config_path = Path(__file__).parent.parent / "config.example.yaml"
        
        if example_config_path.exists():
            import shutil
            shutil.copy(example_config_path, output_path)
            rprint(f"[green]Configuration template created at {output_path}[/green]")
            rprint("[yellow]Please edit the configuration file with your connection details.[/yellow]")
        else:
            rprint("[red]Configuration template not found[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        rprint(f"[red]Error creating configuration: {e}[/red]")
        raise typer.Exit(1)


def display_analysis_results(result):
    """Display analysis results in a formatted way."""
    console.print(Panel.fit(
        f"[bold green]Analysis Complete[/bold green]\n"
        f"Database: {result.request.database}\n"
        f"Connection: {result.request.connection_type.upper()}\n"
        f"Execution Time: {result.execution_time:.2f}s",
        title="Analysis Summary"
    ))
    
    # Schema Analysis Summary
    if result.schema_analysis:
        schema = result.schema_analysis
        summary = schema.get('summary', {})
        
        table = Table(title="Schema Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="bold")
        
        table.add_row("Total Tables", str(summary.get('total_tables', 0)))
        table.add_row("Total Columns", str(summary.get('total_columns', 0)))
        table.add_row("Total Rows", f"{summary.get('total_rows', 0):,}")
        table.add_row("Relationships Detected", str(len(schema.get('relationships', []))))
        
        console.print(table)
    
    # Data Quality Summary
    if result.data_profiles:
        quality_scores = []
        for table_name, profile in result.data_profiles.items():
            if isinstance(profile, dict) and 'overall_quality_score' in profile:
                quality_scores.append(profile['overall_quality_score'])
        
        if quality_scores:
            avg_quality = sum(quality_scores) / len(quality_scores)
            quality_color = "green" if avg_quality > 0.8 else "yellow" if avg_quality > 0.6 else "red"
            
            console.print(Panel.fit(
                f"[bold {quality_color}]Average Data Quality Score: {avg_quality:.2f}[/bold {quality_color}]",
                title="Data Quality Summary"
            ))
    
    # Workflow Insights (if available)
    if hasattr(result, 'workflow_insights') and result.workflow_insights:
        insights = result.workflow_insights
        execution_summary = insights.get('execution_summary', {})
        
        console.print("\n[bold blue]Workflow Execution Summary:[/bold blue]")
        console.print(f"  Stages Executed: {execution_summary.get('successful_stages', 0)}/{execution_summary.get('total_stages', 0)}")
        
        if 'confidence_analysis' in insights:
            conf_analysis = insights['confidence_analysis']
            console.print(f"  Average Confidence: {conf_analysis.get('average_confidence', 0):.2f}")
            console.print(f"  Highest Confidence Stage: {conf_analysis.get('highest_confidence_stage', 'unknown')}")
        
        if 'recommendation_distribution' in insights:
            rec_dist = insights['recommendation_distribution']
            console.print(f"  Total Recommendations: {rec_dist.get('total_recommendations', 0)}")
    
    # AI Recommendations Summary
    if result.ai_recommendations:
        recommendations_by_category = {}
        recommendations_by_source = {}
        
        for rec in result.ai_recommendations:
            category = rec.get('category', 'general')
            source = rec.get('source_stage', 'unknown')
            
            if category not in recommendations_by_category:
                recommendations_by_category[category] = 0
            recommendations_by_category[category] += 1
            
            if source not in recommendations_by_source:
                recommendations_by_source[source] = 0
            recommendations_by_source[source] += 1
        
        # Category breakdown
        table = Table(title="AI Recommendations by Category")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="bold")
        
        for category, count in recommendations_by_category.items():
            table.add_row(category.replace('_', ' ').title(), str(count))
        
        console.print(table)
        
        # Source breakdown (if workflow was used)
        if len(recommendations_by_source) > 1:
            table2 = Table(title="Recommendations by Agent")
            table2.add_column("Agent", style="green")
            table2.add_column("Count", style="bold")
            
            for source, count in recommendations_by_source.items():
                table2.add_row(source.replace('_', ' ').title(), str(count))
            
            console.print(table2)
        
        # Show top recommendations
        high_priority_recs = [
            rec for rec in result.ai_recommendations 
            if rec.get('consolidated_priority', rec.get('priority', 'medium')) in ['critical', 'high']
        ][:5]
        
        if high_priority_recs:
            console.print("\n[bold red]Top Priority Recommendations:[/bold red]")
            for i, rec in enumerate(high_priority_recs, 1):
                priority = rec.get('consolidated_priority', rec.get('priority', 'medium'))
                source = rec.get('source_stage', 'unknown')
                
                console.print(f"{i}. [bold]{rec.get('title', 'Unknown')}[/bold] [{priority.upper()}]")
                console.print(f"   Source: {source.replace('_', ' ').title()}")
                console.print(f"   Category: {rec.get('category', 'general').replace('_', ' ').title()}")
                if rec.get('description'):
                    # Truncate description if too long
                    desc = rec['description'][:150] + "..." if len(rec['description']) > 150 else rec['description']
                    console.print(f"   {desc}")
                console.print()


def save_results(result, output_path: Path, format: str):
    """Save analysis results to file."""
    # Convert result to serializable format
    data = {
        'request': {
            'connection_type': result.request.connection_type,
            'database': result.request.database,
            'tables': result.request.tables,
            'include_data_profiling': result.request.include_data_profiling,
            'include_ai_recommendations': result.request.include_ai_recommendations,
            'sample_size': result.request.sample_size,
            'context': result.request.context
        },
        'success': result.success,
        'execution_time': result.execution_time,
        'schema_analysis': result.schema_analysis,
        'data_profiles': result.data_profiles,
        'ai_recommendations': result.ai_recommendations,
        'metadata': result.metadata
    }
    
    # Add workflow-specific data if available
    if hasattr(result, 'workflow_insights') and result.workflow_insights:
        data['workflow_insights'] = result.workflow_insights
    
    if hasattr(result, 'agent_results') and result.agent_results:
        # Convert agent results to serializable format
        agent_results_serializable = {}
        for stage, agent_response in result.agent_results.items():
            agent_results_serializable[stage.value if hasattr(stage, 'value') else str(stage)] = {
                'success': agent_response.success,
                'recommendations': agent_response.recommendations,
                'confidence': agent_response.confidence,
                'metadata': agent_response.metadata,
                'error_message': agent_response.error_message
            }
        data['agent_results'] = agent_results_serializable
    
    if result.error_message:
        data['error_message'] = result.error_message
    
    # Save in requested format
    if format.lower() == 'yaml':
        with open(output_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, indent=2)
    else:  # Default to JSON
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    app()
