import React, { Component } from 'react';
import {connect, Provider} from 'react-redux';
import { createStore, combineReducers } from 'redux';
import { reducer as reduxFormReducer } from 'redux-form';
import LayoutWrapper from '../../components/utility/layoutWrapper';
import Papersheet from '../../components/utility/papersheet';
import { FullColumn } from '../../components/utility/rowColumn';
import {withStyles} from "@material-ui/core";
import action from "../../redux/ViewApplication/actions";
import {store} from "../../redux/store";


class ViewApplication extends Component {
    state = {
        result: '',
    };

    componentDidMount() {
        store.dispatch(action.requestApplication(this.props.match.params.id));
    }

    render() {
        const { result } = this.state;
        return (
            <LayoutWrapper>
                <FullColumn>
                    <Papersheet>

                    </Papersheet>
                </FullColumn>
            </LayoutWrapper>
        );
    }
}

const mapStateToProps = (state) => {return { ...state.ViewApplication} };
export default connect(mapStateToProps)(ViewApplication);

