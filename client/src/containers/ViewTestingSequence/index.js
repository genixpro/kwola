import React, { Component } from 'react';
import {connect, Provider} from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import PageTitle from '../../components/utility/paperTitle';
import Papersheet, { DemoWrapper } from '../../components/utility/papersheet';
import { FullColumn , HalfColumn, OneThirdColumn, TwoThirdColumn, Row, Column} from '../../components/utility/rowColumn';
import {withStyles} from "@material-ui/core";
import action from "../../redux/ViewTestingSequence/actions";
import {store} from "../../redux/store";
import SingleCard from '../Shuffle/singleCard.js';
import BoxCard from '../../components/boxCard';
import Img7 from '../../images/7.jpg';
import user from '../../images/user.jpg';
import ActionButton from "../../components/mail/singleMail/actionButton";
import {Button} from "../UiElements/Button/button.style";
import Icon from "../../components/uielements/icon";
import moment from 'moment';
import {Chip, Wrapper} from "../UiElements/Chips/chips.style";
import Avatar from "../../components/uielements/avatars";
import {Table} from "../ListApplications/materialUiTables.style";
import {TableBody, TableCell, TableHead, TableRow} from "../../components/uielements/table";

class ViewTestingSequence extends Component {
    state = {
        result: '',
    };

    componentDidMount() {
        store.dispatch(action.requestTestingSequence(this.props.match.params.id));
    }

    // launchTestingSequenceButtonClicked() {
    //     store.dispatch(action.requestNewTestingSequence(this.props.match.params.id));
    // }

    render() {
        const { result } = this.state;
        return (
            this.props.testingSequence ?
                <LayoutWrapper>
                    <FullColumn>
                        <Row>
                            <HalfColumn>
                                <SingleCard src={Img7} grid/>
                            </HalfColumn>

                            <HalfColumn>
                                <Papersheet
                                    title={`${this.props.testingSequence.name}`}
                                    subtitle={`Last Tested ${moment(this.props.timestamp).format('MMM Do, YYYY')}`}
                                >

                                    AWESOME!!!

                                </Papersheet>


                            </HalfColumn>
                        </Row>
                        <Row>
                            <FullColumn>
                                <Papersheet title={"Bug"}>


                                    <Table>
                                        <TableHead>
                                            <TableRow>
                                                <TableCell>ID</TableCell>
                                                <TableCell>Status</TableCell>
                                                <TableCell>Test Start Time</TableCell>
                                                <TableCell>Bug Found</TableCell>
                                            </TableRow>
                                        </TableHead>
                                        <TableBody>

                                            {(this.props.executionSessions || []).map(session => {
                                                return (
                                                    <TableRow key={session._id.$oid} hover={true} onClick={() => this.props.history.push(`/dashboard/execution_sessions/${session._id.$oid}`)}>
                                                        <TableCell>{session._id.$oid}</TableCell>
                                                        <TableCell>{session.status}</TableCell>
                                                        <TableCell>{session.startDate}</TableCell>
                                                        <TableCell>{session.bugsFound}</TableCell>
                                                    </TableRow>
                                                );
                                            })}
                                        </TableBody>
                                    </Table>
                                </Papersheet>

                            </FullColumn>
                        </Row>

                    </FullColumn>
                </LayoutWrapper>
                : null

        );
    }
}

const mapStateToProps = (state) => {return { ...state.ViewTestingSequence} };
export default connect(mapStateToProps)(ViewTestingSequence);

