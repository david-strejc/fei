"""
Mediator for Fei code assistant.

This module contains the Mediator class, which facilitates
communication between components without direct dependencies.
"""

from typing import Dict, List, Optional, Any, Callable, Set

from fei.utils.logging import get_logger

logger = get_logger(__name__)


class Mediator:
    """
    Facilitates communication between components without direct dependencies.
    
    This class implements the Mediator pattern to reduce coupling between components.
    Components can register event handlers and notify other components of events.
    """
    
    def __init__(self):
        """Initialize the mediator."""
        self.components: Dict[str, Any] = {}
        self.event_handlers: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.registered_events: Set[str] = set()
    
    def register_component(self, name: str, component: Any) -> None:
        """
        Register a component with the mediator.
        
        Args:
            name: Component name
            component: Component instance
        """
        self.components[name] = component
        logger.debug(f"Registered component: {name}")
    
    def get_component(self, name: str) -> Optional[Any]:
        """
        Get a component by name.
        
        Args:
            name: Component name
            
        Returns:
            Component instance or None if not found
        """
        return self.components.get(name)
    
    def register_event(self, event_name: str) -> None:
        """
        Register an event type.
        
        Args:
            event_name: Event name
        """
        self.registered_events.add(event_name)
        logger.debug(f"Registered event: {event_name}")
    
    def register_handler(self, event_name: str, handler: Callable[[Dict[str, Any]], None]) -> None:
        """
        Register an event handler.
        
        Args:
            event_name: Event name
            handler: Event handler function
        """
        if event_name not in self.event_handlers:
            self.event_handlers[event_name] = []
        
        self.event_handlers[event_name].append(handler)
        logger.debug(f"Registered handler for event: {event_name}")
        
        # Auto-register event if not already registered
        if event_name not in self.registered_events:
            self.register_event(event_name)
    
    def notify(self, event_name: str, data: Dict[str, Any]) -> None:
        """
        Notify all handlers of an event.
        
        Args:
            event_name: Event name
            data: Event data
        """
        if event_name not in self.registered_events:
            logger.warning(f"Unregistered event: {event_name}")
            # Auto-register for convenience
            self.register_event(event_name)
        
        handlers = self.event_handlers.get(event_name, [])
        for handler in handlers:
            try:
                handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_name}: {e}")
        
        logger.debug(f"Notified {len(handlers)} handlers of event: {event_name}")
    
    def broadcast(self, event_name: str, data: Dict[str, Any]) -> None:
        """
        Broadcast an event to all components.
        
        This is a more general form of notification that doesn't require
        explicit handler registration. Components can implement an
        on_event(event_name, data) method to receive broadcasts.
        
        Args:
            event_name: Event name
            data: Event data
        """
        # First, notify specific handlers
        self.notify(event_name, data)
        
        # Then, broadcast to all components with an on_event method
        for name, component in self.components.items():
            if hasattr(component, 'on_event') and callable(component.on_event):
                try:
                    component.on_event(event_name, data)
                except Exception as e:
                    logger.error(f"Error in component {name} on_event handler for {event_name}: {e}")
        
        logger.debug(f"Broadcasted event {event_name} to all components")
