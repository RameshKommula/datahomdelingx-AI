"""Workflow Engine - Orchestrates AI agents in a structured workflow."""

from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor
import time
from loguru import logger

from .base_agent import BaseAgent, AgentResponse
from .discovery_agent import DiscoveryAgent
from .analysis_agent import AnalysisAgent
from .modeling_advisor import ModelingAdvisor
from .optimization_agent import OptimizationAgent


class WorkflowStage(Enum):
    """Workflow execution stages."""
    DISCOVERY = "discovery"
    ANALYSIS = "analysis"
    MODELING = "modeling"
    OPTIMIZATION = "optimization"


@dataclass
class WorkflowTask:
    """A single task in the workflow."""
    stage: WorkflowStage
    agent: BaseAgent
    dependencies: List[WorkflowStage] = field(default_factory=list)
    parallel_execution: bool = False
    required: bool = True
    timeout_seconds: int = 300
    retry_count: int = 1


@dataclass
class WorkflowResult:
    """Result of workflow execution."""
    success: bool
    execution_time: float
    stage_results: Dict[WorkflowStage, AgentResponse] = field(default_factory=dict)
    workflow_insights: Dict[str, Any] = field(default_factory=dict)
    consolidated_recommendations: List[Dict[str, Any]] = field(default_factory=list)
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class WorkflowEngine:
    """Orchestrates AI agents in a structured workflow."""
    
    def __init__(self, max_parallel_tasks: int = 3):
        self.max_parallel_tasks = max_parallel_tasks
        self.agents = {
            WorkflowStage.DISCOVERY: DiscoveryAgent(),
            WorkflowStage.ANALYSIS: AnalysisAgent(),
            WorkflowStage.MODELING: ModelingAdvisor(),
            WorkflowStage.OPTIMIZATION: OptimizationAgent()
        }
        
        # Define default workflow
        self.default_workflow = [
            WorkflowTask(
                stage=WorkflowStage.DISCOVERY,
                agent=self.agents[WorkflowStage.DISCOVERY],
                dependencies=[],
                required=True
            ),
            WorkflowTask(
                stage=WorkflowStage.ANALYSIS,
                agent=self.agents[WorkflowStage.ANALYSIS],
                dependencies=[WorkflowStage.DISCOVERY],
                required=True
            ),
            WorkflowTask(
                stage=WorkflowStage.MODELING,
                agent=self.agents[WorkflowStage.MODELING],
                dependencies=[WorkflowStage.DISCOVERY, WorkflowStage.ANALYSIS],
                parallel_execution=True,
                required=False
            ),
            WorkflowTask(
                stage=WorkflowStage.OPTIMIZATION,
                agent=self.agents[WorkflowStage.OPTIMIZATION],
                dependencies=[WorkflowStage.DISCOVERY, WorkflowStage.ANALYSIS],
                parallel_execution=True,
                required=False
            )
        ]
    
    def execute_workflow(self, data: Dict[str, Any], 
                        context: Optional[Dict[str, Any]] = None,
                        custom_workflow: Optional[List[WorkflowTask]] = None) -> WorkflowResult:
        """Execute the AI agent workflow."""
        start_time = time.time()
        workflow = custom_workflow or self.default_workflow
        
        logger.info(f"Starting workflow execution with {len(workflow)} stages")
        
        try:
            # Initialize workflow result
            result = WorkflowResult(
                success=True,
                execution_time=0.0,
                metadata={
                    'workflow_stages': len(workflow),
                    'start_time': start_time,
                    'parallel_tasks': self.max_parallel_tasks
                }
            )
            
            # Execute workflow stages
            stage_results = self._execute_workflow_stages(workflow, data, context)
            result.stage_results = stage_results
            
            # Check for required stage failures
            for task in workflow:
                if task.required and task.stage not in stage_results:
                    result.success = False
                    result.error_message = f"Required stage {task.stage.value} failed"
                    break
                elif task.required and not stage_results[task.stage].success:
                    result.success = False
                    result.error_message = f"Required stage {task.stage.value} failed: {stage_results[task.stage].error_message}"
                    break
            
            if result.success:
                # Generate workflow insights
                result.workflow_insights = self._generate_workflow_insights(stage_results)
                
                # Consolidate recommendations
                result.consolidated_recommendations = self._consolidate_recommendations(stage_results)
                
                logger.info(f"Workflow completed successfully with {len(result.consolidated_recommendations)} recommendations")
            
            result.execution_time = time.time() - start_time
            return result
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return WorkflowResult(
                success=False,
                execution_time=time.time() - start_time,
                error_message=str(e),
                metadata={'error_stage': 'workflow_execution'}
            )
    
    def _execute_workflow_stages(self, workflow: List[WorkflowTask], 
                                data: Dict[str, Any], 
                                context: Optional[Dict[str, Any]]) -> Dict[WorkflowStage, AgentResponse]:
        """Execute workflow stages respecting dependencies."""
        stage_results = {}
        executed_stages = set()
        
        # Build execution plan
        execution_plan = self._build_execution_plan(workflow)
        
        for stage_group in execution_plan:
            if len(stage_group) == 1:
                # Sequential execution
                task = stage_group[0]
                logger.info(f"Executing stage: {task.stage.value}")
                
                # Prepare context with previous stage results
                stage_context = self._prepare_stage_context(context, stage_results, task)
                
                # Execute stage
                stage_result = self._execute_single_stage(task, data, stage_context)
                stage_results[task.stage] = stage_result
                executed_stages.add(task.stage)
                
            else:
                # Parallel execution
                logger.info(f"Executing {len(stage_group)} stages in parallel: {[t.stage.value for t in stage_group]}")
                
                parallel_results = self._execute_parallel_stages(stage_group, data, context, stage_results)
                stage_results.update(parallel_results)
                executed_stages.update(parallel_results.keys())
        
        return stage_results
    
    def _build_execution_plan(self, workflow: List[WorkflowTask]) -> List[List[WorkflowTask]]:
        """Build execution plan respecting dependencies and parallel execution flags."""
        execution_plan = []
        remaining_tasks = workflow.copy()
        executed_stages = set()
        
        while remaining_tasks:
            # Find tasks that can be executed (dependencies met)
            ready_tasks = []
            for task in remaining_tasks:
                if all(dep in executed_stages for dep in task.dependencies):
                    ready_tasks.append(task)
            
            if not ready_tasks:
                # Circular dependency or other issue
                logger.error("No ready tasks found - possible circular dependency")
                break
            
            # Group parallel tasks together
            parallel_group = []
            sequential_group = []
            
            for task in ready_tasks:
                if task.parallel_execution and len(parallel_group) < self.max_parallel_tasks:
                    parallel_group.append(task)
                else:
                    sequential_group.append(task)
            
            # Add parallel group first if it exists
            if parallel_group:
                execution_plan.append(parallel_group)
                for task in parallel_group:
                    remaining_tasks.remove(task)
                    executed_stages.add(task.stage)
            
            # Add sequential tasks one by one
            for task in sequential_group:
                execution_plan.append([task])
                remaining_tasks.remove(task)
                executed_stages.add(task.stage)
                break  # Only one sequential task per iteration
        
        return execution_plan
    
    def _prepare_stage_context(self, base_context: Optional[Dict[str, Any]], 
                              stage_results: Dict[WorkflowStage, AgentResponse],
                              current_task: WorkflowTask) -> Dict[str, Any]:
        """Prepare context for a stage including results from dependency stages."""
        context = base_context.copy() if base_context else {}
        
        # Add results from dependency stages
        for dep_stage in current_task.dependencies:
            if dep_stage in stage_results:
                dep_result = stage_results[dep_stage]
                context[f"{dep_stage.value}_findings"] = dep_result.recommendations
                context[f"{dep_stage.value}_metadata"] = dep_result.metadata
                context[f"{dep_stage.value}_confidence"] = dep_result.confidence
        
        # Add workflow context
        context['workflow_stage'] = current_task.stage.value
        context['executed_stages'] = list(stage_results.keys())
        
        return context
    
    def _execute_single_stage(self, task: WorkflowTask, data: Dict[str, Any], 
                             context: Dict[str, Any]) -> AgentResponse:
        """Execute a single workflow stage."""
        try:
            # Execute with timeout and retry
            for attempt in range(task.retry_count + 1):
                try:
                    logger.debug(f"Executing {task.stage.value} (attempt {attempt + 1})")
                    
                    # Execute the agent
                    result = task.agent.analyze(data, context)
                    
                    if result.success:
                        logger.info(f"Stage {task.stage.value} completed successfully")
                        return result
                    else:
                        logger.warning(f"Stage {task.stage.value} failed: {result.error_message}")
                        if attempt < task.retry_count:
                            logger.info(f"Retrying stage {task.stage.value}")
                            continue
                        else:
                            return result
                
                except Exception as e:
                    logger.error(f"Stage {task.stage.value} attempt {attempt + 1} failed: {e}")
                    if attempt < task.retry_count:
                        continue
                    else:
                        return AgentResponse(
                            success=False,
                            recommendations=[],
                            reasoning="",
                            confidence=0.0,
                            metadata={'stage': task.stage.value, 'error': str(e)},
                            error_message=str(e)
                        )
            
        except Exception as e:
            logger.error(f"Stage {task.stage.value} execution failed: {e}")
            return AgentResponse(
                success=False,
                recommendations=[],
                reasoning="",
                confidence=0.0,
                metadata={'stage': task.stage.value, 'error': str(e)},
                error_message=str(e)
            )
    
    def _execute_parallel_stages(self, tasks: List[WorkflowTask], data: Dict[str, Any],
                                base_context: Optional[Dict[str, Any]],
                                stage_results: Dict[WorkflowStage, AgentResponse]) -> Dict[WorkflowStage, AgentResponse]:
        """Execute multiple stages in parallel."""
        results = {}
        
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            # Submit all tasks
            future_to_task = {}
            for task in tasks:
                stage_context = self._prepare_stage_context(base_context, stage_results, task)
                future = executor.submit(self._execute_single_stage, task, data, stage_context)
                future_to_task[future] = task
            
            # Collect results
            for future in future_to_task:
                task = future_to_task[future]
                try:
                    result = future.result(timeout=task.timeout_seconds)
                    results[task.stage] = result
                except Exception as e:
                    logger.error(f"Parallel stage {task.stage.value} failed: {e}")
                    results[task.stage] = AgentResponse(
                        success=False,
                        recommendations=[],
                        reasoning="",
                        confidence=0.0,
                        metadata={'stage': task.stage.value, 'error': str(e)},
                        error_message=str(e)
                    )
        
        return results
    
    def _generate_workflow_insights(self, stage_results: Dict[WorkflowStage, AgentResponse]) -> Dict[str, Any]:
        """Generate insights from the overall workflow execution."""
        insights = {
            'execution_summary': {
                'total_stages': len(stage_results),
                'successful_stages': len([r for r in stage_results.values() if r.success]),
                'failed_stages': len([r for r in stage_results.values() if not r.success])
            },
            'confidence_analysis': {},
            'recommendation_distribution': {},
            'cross_stage_correlations': []
        }
        
        # Confidence analysis
        confidences = {stage.value: result.confidence for stage, result in stage_results.items() if result.success}
        if confidences:
            insights['confidence_analysis'] = {
                'average_confidence': sum(confidences.values()) / len(confidences),
                'stage_confidences': confidences,
                'highest_confidence_stage': max(confidences.items(), key=lambda x: x[1])[0],
                'lowest_confidence_stage': min(confidences.items(), key=lambda x: x[1])[0]
            }
        
        # Recommendation distribution
        rec_counts = {}
        total_recommendations = 0
        
        for stage, result in stage_results.items():
            if result.success:
                count = len(result.recommendations)
                rec_counts[stage.value] = count
                total_recommendations += count
        
        insights['recommendation_distribution'] = {
            'total_recommendations': total_recommendations,
            'by_stage': rec_counts,
            'average_per_stage': total_recommendations / len(rec_counts) if rec_counts else 0
        }
        
        # Cross-stage correlations
        insights['cross_stage_correlations'] = self._find_cross_stage_correlations(stage_results)
        
        return insights
    
    def _find_cross_stage_correlations(self, stage_results: Dict[WorkflowStage, AgentResponse]) -> List[Dict[str, Any]]:
        """Find correlations and connections between stage results."""
        correlations = []
        
        # Compare recommendations across stages
        stage_recommendations = {}
        for stage, result in stage_results.items():
            if result.success:
                stage_recommendations[stage] = result.recommendations
        
        # Look for similar themes or overlapping recommendations
        for stage1, recs1 in stage_recommendations.items():
            for stage2, recs2 in stage_recommendations.items():
                if stage1 != stage2:
                    correlation = self._calculate_recommendation_overlap(recs1, recs2)
                    if correlation['overlap_score'] > 0.3:  # Significant overlap
                        correlations.append({
                            'stage1': stage1.value,
                            'stage2': stage2.value,
                            'overlap_score': correlation['overlap_score'],
                            'common_themes': correlation['common_themes'],
                            'complementary_insights': correlation['complementary_insights']
                        })
        
        return correlations
    
    def _calculate_recommendation_overlap(self, recs1: List[Dict[str, Any]], 
                                        recs2: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate overlap between two sets of recommendations."""
        # Extract categories and keywords from recommendations
        categories1 = set(rec.get('category', 'general') for rec in recs1)
        categories2 = set(rec.get('category', 'general') for rec in recs2)
        
        # Calculate category overlap
        common_categories = categories1.intersection(categories2)
        total_categories = categories1.union(categories2)
        category_overlap = len(common_categories) / len(total_categories) if total_categories else 0
        
        # Extract keywords from titles and descriptions
        keywords1 = self._extract_keywords_from_recommendations(recs1)
        keywords2 = self._extract_keywords_from_recommendations(recs2)
        
        common_keywords = keywords1.intersection(keywords2)
        total_keywords = keywords1.union(keywords2)
        keyword_overlap = len(common_keywords) / len(total_keywords) if total_keywords else 0
        
        # Overall overlap score
        overlap_score = (category_overlap + keyword_overlap) / 2
        
        return {
            'overlap_score': overlap_score,
            'common_themes': list(common_categories),
            'complementary_insights': {
                'unique_to_first': list(categories1 - categories2),
                'unique_to_second': list(categories2 - categories1)
            }
        }
    
    def _extract_keywords_from_recommendations(self, recommendations: List[Dict[str, Any]]) -> set:
        """Extract keywords from recommendation titles and descriptions."""
        keywords = set()
        
        for rec in recommendations:
            title = rec.get('title', '').lower()
            description = rec.get('description', '').lower()
            
            # Simple keyword extraction (can be enhanced with NLP)
            text = title + ' ' + description
            words = text.split()
            
            # Filter for meaningful words (basic approach)
            meaningful_words = [word for word in words 
                             if len(word) > 3 and 
                             word not in ['the', 'and', 'for', 'with', 'this', 'that', 'should', 'could', 'would']]
            
            keywords.update(meaningful_words[:10])  # Limit to top 10 words per recommendation
        
        return keywords
    
    def _consolidate_recommendations(self, stage_results: Dict[WorkflowStage, AgentResponse]) -> List[Dict[str, Any]]:
        """Consolidate recommendations from all stages into a prioritized list."""
        all_recommendations = []
        
        # Collect all recommendations with stage information
        for stage, result in stage_results.items():
            if result.success:
                for rec in result.recommendations:
                    # Add stage metadata
                    consolidated_rec = rec.copy()
                    consolidated_rec['source_stage'] = stage.value
                    consolidated_rec['stage_confidence'] = result.confidence
                    
                    # Calculate consolidated priority
                    consolidated_rec['consolidated_priority'] = self._calculate_consolidated_priority(
                        rec, stage, result.confidence
                    )
                    
                    all_recommendations.append(consolidated_rec)
        
        # Remove duplicates and merge similar recommendations
        deduplicated_recommendations = self._deduplicate_recommendations(all_recommendations)
        
        # Sort by consolidated priority
        prioritized_recommendations = sorted(
            deduplicated_recommendations,
            key=lambda x: self._priority_sort_key(x['consolidated_priority']),
            reverse=True
        )
        
        return prioritized_recommendations
    
    def _calculate_consolidated_priority(self, recommendation: Dict[str, Any], 
                                       stage: WorkflowStage, stage_confidence: float) -> str:
        """Calculate consolidated priority considering stage and confidence."""
        base_priority = recommendation.get('priority', 'medium')
        
        # Stage-specific priority adjustments
        stage_weights = {
            WorkflowStage.DISCOVERY: 0.8,  # Discovery findings are foundational
            WorkflowStage.ANALYSIS: 1.0,   # Analysis findings are direct
            WorkflowStage.MODELING: 0.9,   # Modeling recommendations are important
            WorkflowStage.OPTIMIZATION: 1.1  # Optimization often has immediate impact
        }
        
        # Priority scoring
        priority_scores = {'critical': 5, 'high': 4, 'medium': 3, 'low': 2}
        base_score = priority_scores.get(base_priority, 3)
        
        # Apply stage weight and confidence
        adjusted_score = base_score * stage_weights.get(stage, 1.0) * stage_confidence
        
        # Convert back to priority levels
        if adjusted_score >= 4.5:
            return 'critical'
        elif adjusted_score >= 3.5:
            return 'high'
        elif adjusted_score >= 2.5:
            return 'medium'
        else:
            return 'low'
    
    def _deduplicate_recommendations(self, recommendations: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate recommendations and merge similar ones."""
        # Simple deduplication based on title similarity
        deduplicated = []
        processed_titles = set()
        
        for rec in recommendations:
            title = rec.get('title', '').lower()
            
            # Check for similar titles (basic approach)
            is_duplicate = False
            for processed_title in processed_titles:
                if self._calculate_title_similarity(title, processed_title) > 0.8:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(rec)
                processed_titles.add(title)
        
        return deduplicated
    
    def _calculate_title_similarity(self, title1: str, title2: str) -> float:
        """Calculate similarity between two titles."""
        words1 = set(title1.split())
        words2 = set(title2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union) if union else 0.0
    
    def _priority_sort_key(self, priority: str) -> int:
        """Convert priority string to sort key."""
        priority_values = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        return priority_values.get(priority, 0)
    
    def create_custom_workflow(self, stages: List[str], 
                              parallel_stages: Optional[List[str]] = None) -> List[WorkflowTask]:
        """Create a custom workflow with specified stages."""
        parallel_stages = parallel_stages or []
        
        workflow = []
        dependencies = []
        
        for i, stage_name in enumerate(stages):
            try:
                stage = WorkflowStage(stage_name)
                agent = self.agents[stage]
                
                task = WorkflowTask(
                    stage=stage,
                    agent=agent,
                    dependencies=dependencies.copy(),
                    parallel_execution=stage_name in parallel_stages,
                    required=True
                )
                
                workflow.append(task)
                dependencies.append(stage)
                
            except ValueError:
                logger.warning(f"Unknown stage: {stage_name}")
                continue
        
        return workflow
