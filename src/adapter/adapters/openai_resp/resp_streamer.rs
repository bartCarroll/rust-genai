use crate::adapter::adapters::support::{StreamerCapturedData, StreamerOptions};
use crate::adapter::inter_stream::{InterStreamEnd, InterStreamEvent};
use crate::chat::ChatOptionsSet;
use crate::{Error, ModelIden, Result};
use reqwest_eventsource::{Event, EventSource};
use serde_json::Value;
use std::pin::Pin;
use std::task::{Context, Poll};
use value_ext::JsonValueExt;

pub struct OpenAIRespStreamer {
	inner: EventSource,
	options: StreamerOptions,

	// -- Set by the poll_next
	/// Flag to prevent polling the EventSource after a completion event
	done: bool,
	captured_data: StreamerCapturedData,
}

impl OpenAIRespStreamer {
	pub fn new(inner: EventSource, model_iden: ModelIden, options_set: ChatOptionsSet<'_, '_>) -> Self {
		Self {
			inner,
			done: false,
			options: StreamerOptions::new(model_iden, options_set),
			captured_data: Default::default(),
		}
	}
}

impl futures::Stream for OpenAIRespStreamer {
	type Item = Result<InterStreamEvent>;

	fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
		if self.done {
			// The last poll was definitely the end, so end the stream.
			// This will prevent triggering a stream ended error
			return Poll::Ready(None);
		}
		
		while let Poll::Ready(event) = Pin::new(&mut self.inner).poll_next(cx) {
			match event {
				Some(Ok(Event::Open)) => return Poll::Ready(Some(Ok(InterStreamEvent::Start))),
				Some(Ok(Event::Message(message))) => {
					// Parse the event data as JSON
					let event_data: Value = serde_json::from_str(&message.data)
						.map_err(|serde_error| Error::StreamParse {
							model_iden: self.options.model_iden.clone(),
							serde_error,
						})?;

					// Get the event type
					let event_type = event_data
						.get("type")
						.and_then(|v| v.as_str())
						.unwrap_or("");

					match event_type {
						// -- response.created - Initial event
						"response.created" => {
							// Just continue, we already sent Start on Event::Open
							continue;
						}

						// -- response.output_text.delta - Text chunk
						"response.output_text.delta" => {
							if let Ok(delta) = event_data.x_get::<String>("/delta") {
								// Capture if enabled
								if self.options.capture_content {
									match self.captured_data.content {
										Some(ref mut c) => c.push_str(&delta),
										None => self.captured_data.content = Some(delta.clone()),
									}
								}

								return Poll::Ready(Some(Ok(InterStreamEvent::Chunk(delta))));
							}
							continue;
						}

						// -- response.output_text.done - Text complete
						"response.output_text.done" => {
							// Text output complete, but might have more items
							continue;
						}

						// -- response.function_call_arguments.delta - Tool call arguments streaming
						"response.function_call_arguments.delta" => {
							// Extract the delta and call_id
							let delta = event_data.x_get::<String>("/delta").unwrap_or_default();
							let call_id = event_data.x_get::<String>("/call_id").unwrap_or_default();
							let name = event_data.x_get::<String>("/name").unwrap_or_default();

							// Accumulate tool call arguments if capturing
							if self.options.capture_tool_calls {
								match &mut self.captured_data.tool_calls {
									Some(calls) => {
										// Find existing call or create new one
										if let Some(existing) = calls.iter_mut().find(|c| c.call_id == call_id) {
											// Accumulate arguments as string
											if let Some(existing_args) = existing.fn_arguments.as_str() {
												existing.fn_arguments = serde_json::Value::String(
													format!("{}{}", existing_args, delta)
												);
											} else {
												existing.fn_arguments = serde_json::Value::String(delta.clone());
											}
										} else {
											// New tool call
											calls.push(crate::chat::ToolCall {
												call_id: call_id.clone(),
												fn_name: name.clone(),
												fn_arguments: serde_json::Value::String(delta.clone()),
											});
										}
									}
									None => {
										self.captured_data.tool_calls = Some(vec![crate::chat::ToolCall {
											call_id: call_id.clone(),
											fn_name: name.clone(),
											fn_arguments: serde_json::Value::String(delta.clone()),
										}]);
									}
								}
							}

							// Return the tool call chunk
							let tool_call = crate::chat::ToolCall {
								call_id,
								fn_name: name,
								fn_arguments: serde_json::Value::String(delta),
							};
							return Poll::Ready(Some(Ok(InterStreamEvent::ToolCallChunk(tool_call))));
						}

						// -- response.function_call_arguments.done - Tool call complete
						"response.function_call_arguments.done" => {
							// Tool call complete, continue
							continue;
						}

						// -- response.completed - Final event
						"response.completed" | "response.done" => {
							self.done = true;

							// Extract usage if available and capturing
							let captured_usage = if self.options.capture_usage {
								event_data
									.x_get::<Value>("/usage")
									.ok()
									.and_then(|usage_val| {
										// Parse the usage into our Usage struct
										serde_json::from_value(usage_val).ok()
									})
							} else {
								None
							};

							// Parse accumulated tool calls into proper JSON
							if let Some(ref mut calls) = self.captured_data.tool_calls {
								for call in calls.iter_mut() {
									if let Some(args_str) = call.fn_arguments.as_str() {
										// Try to parse the accumulated string as JSON
										if let Ok(parsed) = serde_json::from_str::<Value>(args_str) {
											call.fn_arguments = parsed;
										}
									}
								}
							}

							let inter_stream_end = InterStreamEnd {
								captured_usage,
								captured_text_content: self.captured_data.content.take(),
								captured_reasoning_content: self.captured_data.reasoning_content.take(),
								captured_tool_calls: self.captured_data.tool_calls.take(),
							};

							return Poll::Ready(Some(Ok(InterStreamEvent::End(inter_stream_end))));
						}

						// -- response.failed or error
						"response.failed" | "error" => {
							self.done = true;

							return Poll::Ready(Some(Err(Error::ChatResponse {
								model_iden: self.options.model_iden.clone(),
								body: event_data,
							})));
						}

						// -- Unhandled event types - just continue
						_ => {
							// Log or debug unhandled event types if needed
							continue;
						}
					}
				}
				Some(Err(err)) => {
					self.done = true;
					return Poll::Ready(Some(Err(Error::ReqwestEventSource(Box::new(err)))));
				}
				None => {
					// Stream ended naturally
					return Poll::Ready(None);
				}
			}
		}

		Poll::Pending
	}
}
